namespace LmgRecap

open CsvHelper
open CsvHelper.Configuration
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open SixLabors.ImageSharp
open SixLabors.ImageSharp.PixelFormats
open SixLabors.ImageSharp.Processing
open SixLabors.ImageSharp.Processing.Processors.Transforms
open System
open System.IO
open System.Linq
open System.Text.RegularExpressions

type LabelData =
    { TagNames: string[]
      RatingIndexes: int[]
      GeneralIndexes: int[]
      CharacterIndexes: int[] }

type PredictionResult =
    { GeneralTags: (string * float)[]
      RatingTags: (string * float)[]
      CharacterTags: (string * float)[] }

type SelectedTag =
    { TagId: int
      Name: string
      Category: int
      Count: int }

module ArrayMath =
    let argsort (arr: float[]) =
        Array.mapi (fun i x -> i, x) arr |> Array.sortBy snd |> Array.map fst

    let argmax (arr: float[]) = argsort arr |> Array.last

    let diff (arr: float[]) =
        Array.map2 (-) arr[.. arr.Length - 2] arr[1..]

type WaifuDiffusionPredictor(modelPath: string, labelPath: string, useCuda: bool) =
    let kaomojis =
        [ "0_0"
          "(o)_(o)"
          "+_+"
          "+_-"
          "._."
          "<o>_<o>"
          "<|>_<|>"
          "=_="
          ">_<"
          "3_3"
          "6_9"
          ">_o"
          "@_@"
          "^_^"
          "o_o"
          "u_u"
          "x_x"
          "|_|"
          "||_||" ]
        |> set

    let mcutThreshold (probs: float[]) : float =
        let sorted = probs |> Array.sortBy (~-)
        let difs = ArrayMath.diff sorted
        let t = ArrayMath.argmax difs
        (sorted[t] + sorted[t + 1]) / 2.0

    let loadLabels (csvPath: string) : LabelData =
        use reader = new StreamReader(csvPath)
        let config = new CsvConfiguration(Globalization.CultureInfo.InvariantCulture)
        config.PrepareHeaderForMatch <- fun header -> Regex.Replace(header.Header, "_", "").ToLower()
        use csv = new CsvReader(reader, config)
        let records = csv.GetRecords<SelectedTag>().ToArray()

        let applyNamespace (name: string) (category: int) =
            match category with
            | 4 -> "character:" + name
            | 9 -> "rating:" + name
            | _ -> name

        let formatName (name: string) =
            if kaomojis.Contains(name) then
                name
            else
                name.Replace("_", " ")

        let tagNames =
            records
            |> Array.map (fun r ->
                let formattedName = formatName r.Name
                applyNamespace formattedName r.Category)

        let indexesByCat cat =
            records
            |> Array.mapi (fun i r -> i, r.Category)
            |> Array.choose (fun (i, c) -> if c = cat then Some i else None)

        { TagNames = tagNames
          RatingIndexes = indexesByCat 9
          GeneralIndexes = indexesByCat 0
          CharacterIndexes = indexesByCat 4 }

    let buildInferenceSession path =
        use sessionOptions = new SessionOptions()

        if useCuda then
            try
                sessionOptions.AppendExecutionProvider_CUDA()
            with ex ->
                sessionOptions.AppendExecutionProvider_CPU()
        else
            sessionOptions.AppendExecutionProvider_CPU()

        new InferenceSession(modelPath, sessionOptions)

    let session = buildInferenceSession modelPath
    let labelData = loadLabels labelPath
    let modelTargetSize = session.InputMetadata.First().Value.Dimensions[1]

    let prepareImage (imageBytes: byte array) =
        use stream = new MemoryStream(imageBytes)
        use image = Image.Load<Rgb24>(stream)

        image.Mutate(fun ctx ->
            ctx
                .BackgroundColor(Color.White)
                .Resize(
                    ResizeOptions(
                        Size = Size(modelTargetSize, modelTargetSize),
                        Sampler = BicubicResampler(),
                        Mode = ResizeMode.Pad,
                        PadColor = Color.White
                    )
                )
            |> ignore)

        let tensor = DenseTensor<float32>([| 1; modelTargetSize; modelTargetSize; 3 |])

        for y = 0 to modelTargetSize - 1 do
            for x = 0 to modelTargetSize - 1 do
                let pixel = image[x, y]
                tensor[0, y, x, 0] <- float32 pixel.B
                tensor[0, y, x, 1] <- float32 pixel.G
                tensor[0, y, x, 2] <- float32 pixel.R

        tensor

    interface IDisposable with
        member _.Dispose() = session.Dispose()

    member _.predict
        (image: byte array)
        (generalThresh: float)
        (generalMcutEnabled: bool)
        (characterThresh: float)
        (characterMcutEnabled: bool)
        : PredictionResult =

        let inputTensor = prepareImage image
        let inputName = session.InputMetadata |> Seq.head |> (fun x -> x.Key)
        let outputName = session.OutputMetadata |> Seq.head |> (fun x -> x.Key)

        let inputs = [| NamedOnnxValue.CreateFromTensor(inputName, inputTensor) |]

        let results = session.Run(inputs)
        let output = results[0].AsTensor<float32>().ToArray() |> Array.map float

        let getTags indices probsThresh useMcut =
            let probs = indices |> Array.map (fun i -> output[i])

            let thresh =
                if useMcut then
                    max 0.15 (mcutThreshold probs)
                else
                    probsThresh

            indices
            |> Array.mapi (fun j idx -> labelData.TagNames[idx], probs[j])
            |> Array.filter (fun (_, p) -> p > thresh)
            |> Array.sortByDescending (fun (_, p) -> p)

        let generalTags = getTags labelData.GeneralIndexes generalThresh generalMcutEnabled

        let ratingTags =
            labelData.RatingIndexes
            |> Array.map (fun i -> labelData.TagNames[i], output[i])
            |> Array.maxBy snd

        let characterTags =
            getTags labelData.CharacterIndexes characterThresh characterMcutEnabled

        { GeneralTags = generalTags
          RatingTags = [| ratingTags |]
          CharacterTags = characterTags }
