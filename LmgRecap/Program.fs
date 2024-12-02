open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Linq
open System.Net.Http
open System.Reflection
open System.Text
open System.Text.Json
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks
open System.Web
open FSharp.Control
open FSUtils
open Logging
open Microsoft.Extensions.Configuration
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.ML.OnnxRuntimeGenAI
open Microsoft.SemanticKernel
open Microsoft.SemanticKernel.Connectors.Chroma
open Microsoft.SemanticKernel.Connectors.OpenAI
open Microsoft.SemanticKernel.Memory
open Microsoft.SemanticKernel.Plugins.Memory
open OpenAI
open OpenQA.Selenium
open OpenQA.Selenium.Firefox
open Python.Runtime
open SixLabors.ImageSharp
open SixLabors.ImageSharp.PixelFormats
open SixLabors.ImageSharp.Processing
open YamlDotNet.Core
open YamlDotNet.Serialization
open YamlDotNet.Serialization.NamingConventions

type Selenium =
    { Downloads: string
      Profile: string
      UserAgent: string }

type Service =
    { Endpoint: string
      Key: string
      Model: string }

type LogLevelConfig =
    { Default: Microsoft.Extensions.Logging.LogLevel }

type LoggingConfig = { LogLevel: LogLevelConfig }

type AppSettings =
    { Embeddings: Service
      Multimodal: Service
      Completion: Service
      MemoryStore: Service
      ResnetModelPath: string
      MinimumRating: int
      MinimumChainRating: int
      MaxReplies: int
      MaxLength: int
      RateMultiple: bool
      RateChain: bool
      Describe: bool
      Selenium: Selenium
      Filters: string[]
      Logging: LoggingConfig }

type MyRedirectingHandler(appSettings) =
    inherit DelegatingHandler(new HttpClientHandler())

    override this.SendAsync
        (
            request: HttpRequestMessage,
            cancellationToken: CancellationToken
        ) : Task<HttpResponseMessage> =
        let embeddingsUri = new Uri(appSettings.Embeddings.Endpoint)
        let uriBuilder = new UriBuilder(request.RequestUri)
        uriBuilder.Host <- embeddingsUri.Host
        uriBuilder.Port <- embeddingsUri.Port
        uriBuilder.Scheme <- embeddingsUri.Scheme
        request.RequestUri <- uriBuilder.Uri
        base.SendAsync(request, cancellationToken)

type Post =
    { no: int64
      com: Option<string>
      tim: Option<int64>
      ext: Option<string>
      time: int64
      now: string }

type Thread = { posts: Post[] }

type CaptionResponse = { content: string }

type Message = { content: string }

type Choice = { message: Message }

type CaptionResponse2 = { choices: Choice[] }

type ChainNodeViewModel =
    { id: int64
      comment: string
      attachment: string
      context: string }

[<CLIMutable>]
type RatingOutput =
    { KeyFactors: string
      Analysis: string
      Rating: int }

[<CLIMutable>]
type MultiRateOutput =
    { Id: int64
      KeyFactors: string
      Analysis: string
      Rating: int }

[<CLIMutable>]
type DescribeOutput =
    { Clues: string
      Reasoning: string
      Summary: string }

type ChainNode =
    { id: int64
      timestamp: int64
      now: string
      filtered: bool
      links: int64[]
      replies: int64[]
      mutable ratings: RatingOutput[]
      comment: string
      unsanitized: string
      context: Option<string>
      mutable rating: int
      reasoning: string
      filename: Option<string>
      caption: Option<string>
      label: Option<string>
      confidence: Option<float32> }

type Chain =
    { mutable Nodes: ChainNode[]
      Category: string
      Rating: int
      Ratings: RatingOutput[]
      Summary: string }

type RecapBuilder =
    { ThreadId: string
      Chains: Chain[]
      Recaps: Chain[] }

let createRecapBuilder thread =
    { ThreadId = thread
      Chains = [||]
      Recaps = [||] }

let toViewModel node =
    { id = node.id
      comment =
        if String.IsNullOrEmpty(node.comment) then
            "None."
        else
            node.comment
      attachment = defaultArg node.caption "None."
      context = defaultArg node.context "None." }

let prettyPrintViewModel nodes =
    SerializerBuilder()
        .WithDefaultScalarStyle(ScalarStyle.Literal)
        .WithNamingConvention(PascalCaseNamingConvention.Instance)
        .Build()
        .Serialize(nodes)
        .ReplaceLineEndings("\n")
        .Replace("\n\n", "\n")
        .Replace("\n\n", "\n")
        .Trim()

let deserializer =
    DeserializerBuilder()
        .WithNamingConvention(PascalCaseNamingConvention.Instance)
        .Build()

let mutable appSettings =
    (new ConfigurationBuilder())
        .AddJsonFile("appsettings.json")
        .AddUserSecrets(Assembly.GetExecutingAssembly())
        .Build()
        .Get<AppSettings>()

let mutable globalLogger = Unchecked.defaultof<ILogger>

let tryWrapper (f: 'b -> 'a) (b: 'b) : Result<'a, string> =
    try
        f b |> Ok
    with e ->
        globalLogger.LogError e.Message
        Error e.Message

let session = new InferenceSession(appSettings.ResnetModelPath)

let imageToOnnx (imageBytes: byte array) (size: int) =
    let stream = new MemoryStream(imageBytes)
    use image = Image.Load<Rgb24>(stream)
    image.Mutate(fun c -> c.Resize(size, size) |> ignore)
    let tensor = new DenseTensor<float32>([| 1; 3; size; size |])

    for y = 0 to size - 1 do
        for x = 0 to size - 1 do
            let pixel = image[x, y]
            tensor[0, 0, y, x] <- float32 pixel.R / 255.0f
            tensor[0, 1, y, x] <- float32 pixel.G / 255.0f
            tensor[0, 2, y, x] <- float32 pixel.B / 255.0f

    tensor

let identify imageBytes =
    let labels = [| "miku"; "other"; "teto" |]
    let tensor = imageToOnnx imageBytes 192
    let inputs = [ NamedOnnxValue.CreateFromTensor("tensor", tensor) ]

    let results = session.Run(inputs)

    let outputTensor = results[0].AsTensor<float32>()
    let maxScore = outputTensor |> Seq.max
    let maxIndex = Array.IndexOf(outputTensor |> Seq.toArray, maxScore)

    (labels[maxIndex], maxScore)

let memories =
    let myHandler = new MyRedirectingHandler(appSettings)
    let client = new HttpClient(myHandler)

    MemoryBuilder()
        .WithOpenAITextEmbeddingGeneration(appSettings.Embeddings.Model, appSettings.Embeddings.Key, null, client)
        .WithMemoryStore(
            new ChromaMemoryStore(
                appSettings.MemoryStore.Endpoint,
                (loggerFactory appSettings.Logging.LogLevel.Default)
            )
        )
        .Build()

let kernel =
    let aiClient =
        let clientOptions = new OpenAIClientOptions()
        clientOptions.Endpoint <- new Uri(appSettings.Completion.Endpoint)

        new OpenAIClient(new ClientModel.ApiKeyCredential(appSettings.Completion.Key), clientOptions)

    let builder = Kernel.CreateBuilder()

    builder.Services.AddSingleton((loggerFactory appSettings.Logging.LogLevel.Default))
    |> ignore

    builder.AddOpenAIChatCompletion(appSettings.Completion.Model, aiClient)
    |> ignore

    builder.Build()

let recapPluginDirectoryPath =
    Path.Combine(Directory.GetCurrentDirectory(), "plugins", "RecapPlugin")

let recapPluginFunctions =
    kernel.ImportPluginFromPromptDirectory(recapPluginDirectoryPath)

let recapMemoryPlugin =
    kernel.ImportPluginFromObject(new TextMemoryPlugin(memories))

let importMarkdownFileIntoMemories filename =
    Runtime.PythonDLL <- "python311.dll"
    PythonEngine.Initialize()
    use _ = Py.GIL()
    use scope = Py.CreateScope()
    scope.Exec(File.ReadAllText("RecreateChromaDb.py")) |> ignore

    File.ReadAllText(filename)
    |> fun t -> t.Split("\r\n\r\n")
    |> Array.map (fun s -> s.Replace("\r\n- ", " ").Replace("**", ""))
    |> Array.iter (fun l ->
        (memories.SaveInformationAsync("Glossary", l, Guid.NewGuid().ToString()).Result
         |> ignore))

let ask kernelFunction (kernelArguments: KernelArguments) =
    for key in kernelArguments.Names do
        globalLogger.LogDebug(key + ":\n" + kernelArguments[key].ToString())

    kernelArguments
    |> set TextMemoryPlugin.CollectionParam "Glossary"
    |> set TextMemoryPlugin.LimitParam "1"
    |> set TextMemoryPlugin.RelevanceParam "0.8"
    |> fun args ->
        kernel
            .InvokeAsync(kernelFunction, args)
            .Result.ToString()
            .Replace("```yaml", "")
            .Replace("```", "")
            .Replace("Unrated:", "")
            .Trim()
    |> fun i ->
        globalLogger.LogDebug("Response:\n" + i)
        i

let askDefault kernelFunction prompt =
    KernelArguments() |> set "input" prompt |> ask kernelFunction

let loadRecapFromSaveFile builder =
    let filename = $"{builder.ThreadId}.recap.json"

    match File.Exists(filename) with
    | false -> builder
    | true -> filename |> File.ReadAllText |> JsonSerializer.Deserialize<RecapBuilder>

let options = new FirefoxOptions()
options.AddArgument("-headless")
options.Profile <- new FirefoxProfile(appSettings.Selenium.Profile)
Directory.CreateDirectory(appSettings.Selenium.Downloads) |> ignore
options.SetPreference("browser.download.dir", appSettings.Selenium.Downloads)
options.SetPreference("browser.download.folderList", 2)
options.SetPreference("browser.download.manager.showWhenStarting", false)
options.SetPreference("browser.helperApps.neverAsk.saveToDisk", "image/jpeg")
options.SetPreference("browser.helperApps.neverAsk.saveToDisk", "image/png")
options.SetPreference("browser.helperApps.neverAsk.saveToDisk", "image/gif")
options.SetPreference("devtools.jsonview.enabled", false)
options.SetPreference("general.useragent.override", appSettings.Selenium.UserAgent)

let mapPostToChainNode post =
    { id = post.no
      timestamp = post.time
      now = post.now
      filtered = appSettings.Filters.Any(fun filter -> Regex.IsMatch((defaultArg post.com "").ToLower(), filter))
      links = [||]
      replies = [||]
      ratings = [||]
      comment = ""
      unsanitized = defaultArg post.com ""
      context = None
      rating = -1
      reasoning = ""
      filename =
        match post.tim.IsSome && post.ext.IsSome with
        | true -> Some(post.tim.Value.ToString() + post.ext.Value.ToString())
        | _ -> None
      caption = None
      label = None
      confidence = None }

let buildReferences (chainmap: Dictionary<int64, ChainNode>) node =
    { node with
        links =
            Regex.Matches(node.unsanitized, @"&gt;&gt;(\d{9})")
            |> Seq.map (fun m -> m.Groups[1].ToString() |> Int64.Parse)
            |> Seq.filter (fun id -> chainmap.ContainsKey id)
            |> Array.ofSeq
        replies =
            chainmap.Values
            |> Seq.filter (fun p -> p.unsanitized.Contains(node.id.ToString()))
            |> Seq.map (fun p -> p.id)
            |> Seq.toArray }

let sanitize (driver: FirefoxDriver) node =
    let fetchTitleFromUrlAsync (url: string) () : string =
        driver.Navigate().GoToUrl url
        Thread.Sleep(10 * 1000)
        driver.Title

    let stripTags (input: string) =
        let sb = StringBuilder()
        let mutable inTag = false

        for c in input do
            match c with
            | '<' -> inTag <- true
            | '>' -> inTag <- false
            | _ when not inTag -> sb.Append(c) |> ignore
            | _ -> ()

        sb.ToString()

    match node.comment = String.Empty with
    | true ->
        { node with
            comment =
                node.unsanitized.Replace("<br>", "\n")
                |> stripTags
                |> HttpUtility.HtmlDecode
                |> fun c ->
                    Regex
                        .Replace(c, @">>\d{9}", String.Empty)
                        .Replace("https://arxiv.org/pdf", "https://arxiv.org/abs")
                        .Trim()
                |> fun c ->
                    (Regex.Matches(
                        c,
                        @"https:\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?"
                     )
                     |> Seq.map (fun m ->
                         m.Value,
                         tryWrapper (fetchTitleFromUrlAsync m.Value) ()
                         |> Result.defaultValue "Error fetching title")
                     |> Seq.fold (fun (newText: string) (url, title) -> newText.Replace(url, $"[{title}]({url})")) c) }
    | _ -> node

let sortNodesIntoRecap (builder: RecapBuilder) (chainNodes: ChainNode[]) =
    let mutable chains = builder.Chains
    let mutable recaps = builder.Recaps

    chainNodes
    |> Array.iter (fun node ->
        let linkedNode =
            node.links
            |> Seq.tryPick (fun linkId ->
                chains
                |> Seq.tryPick (fun chain -> chain.Nodes |> Seq.tryFind (fun n -> n.id = linkId)))

        match linkedNode with
        | Some linkedChainNode ->
            let targetChain =
                chains |> Seq.find (fun chain -> chain.Nodes.Contains linkedChainNode)

            targetChain.Nodes <- Array.append targetChain.Nodes [| node |]
        | None ->
            match node.comment.Contains("Recent Highlights from the Previous Thread") with
            | true ->
                recaps <-
                    Array.append
                        recaps
                        [| { Nodes = [| node |]
                             Category = ""
                             Rating = -1
                             Ratings = [||]
                             Summary = "" } |]
            | false ->
                chains <-
                    Array.append
                        chains
                        [| { Nodes = [| node |]
                             Category = ""
                             Rating = -1
                             Ratings = [||]
                             Summary = "" } |])

    { builder with
        Chains = chains
        Recaps = recaps }

let fetchThreadJson (driver: FirefoxDriver) builder =
    let cnm = new Dictionary<int64, ChainNode>()

    let populateChainNodeDictionary s =
        for i in s do
            cnm.Add(i.id, i)

        s

    builder.Chains
    |> Seq.collect (fun c -> c.Nodes)
    |> populateChainNodeDictionary
    |> ignore

    let maxId =
        match Seq.isEmpty builder.Chains with
        | true -> 0L
        | false -> cnm.Values |> Seq.maxBy (fun n -> n.id) |> (fun n -> n.id)

    driver
        .Navigate()
        .GoToUrl($"https://a.4cdn.org/g/thread/{builder.ThreadId}.json")

    let updateReferences oldbuilder =
        { oldbuilder with
            Chains =
                oldbuilder.Chains
                |> Array.map (fun chain ->
                    { chain with
                        Nodes = chain.Nodes |> Array.map (buildReferences cnm) }) }

    driver.FindElement(By.TagName("html")).Text
    |> JsonSerializer.Deserialize<Thread>
    |> fun a -> a.posts[1..]
    |> Array.filter (fun p -> p.no > maxId)
    |> Array.map mapPostToChainNode
    |> Array.map (sanitize driver)
    |> populateChainNodeDictionary
    |> Array.map (buildReferences cnm)
    |> sortNodesIntoRecap builder
    |> updateReferences

let recall node =
    match Option.isNone node.context with
    | true ->
        try
            memories
                .SearchAsync("Glossary", node.comment, 1, 0.8)
                .FirstOrDefaultAsync()
                .AsTask()
            |> Task.toAsync
            |> Async.RunSynchronously
            |> fun t ->
                { node with
                    context = Some(t.Metadata.Text.ToString()) }
        with _ ->
            { node with context = Some "None." }
    | _ -> node

type Phi3Model =
    { Path: string
      SystemPrompt: string
      UserPrompt: string
      FullPrompt: string
      Model: Model
      Processor: MultiModalProcessor
      TokenizerStream: TokenizerStream }

    static member New() =
        let path = @"D:\_models\Phi-3-vision-128k-instruct-onnx-cuda\cuda-int4-rtn-block-32"

        let systemPrompt =
            "You are an AI assistant that helps people find information. Answer questions using a direct style."

        let userPrompt =
            "Describe what is in the image. If the image is a chatlog with an AI chatbot, either as an assistant or a character in a roleplay, use the word chatlog in your description. Otherwise don't mention it."

        let model = new Model(path)
        let processor = new MultiModalProcessor(model)

        { Path = path
          SystemPrompt = systemPrompt
          UserPrompt = userPrompt
          FullPrompt = $"<|system|>{systemPrompt}<|end|><|user|><|image_1|>{userPrompt}<|end|><|assistant|>"
          Model = model
          Processor = processor
          TokenizerStream = processor.CreateStream() }

let captionNodePhi3 phi3Model imagePath =
    let img = Images.Load imagePath

    globalLogger.LogInformation "Start processing image and prompt ..."
    let inputTensors = phi3Model.Processor.ProcessImages(phi3Model.FullPrompt, img)

    use generatorParams = new GeneratorParams(phi3Model.Model)
    generatorParams.SetSearchOption("max_length", 3072)
    generatorParams.SetInputs(inputTensors)

    globalLogger.LogInformation "Generating response ..."
    let mutable response = new StringBuilder()
    use generator = new Generator(phi3Model.Model, generatorParams)

    while not (generator.IsDone()) do
        generator.ComputeLogits()
        generator.GenerateNextToken()
        let seq = generator.GetSequence(0UL)
        let lastElement = phi3Model.TokenizerStream.Decode seq[seq.Length - 1]
        response <- response.Append(lastElement)

    globalLogger.LogInformation("Generation complete: {GeneratorResponse}", response)
    response.ToString().Trim()

let captionNode (driver: FirefoxDriver) phi3Model node =
    let mutable caption = node.caption

    Directory.EnumerateFiles(appSettings.Selenium.Downloads)
    |> Seq.iter (fun f -> File.Delete(f))

    let downloadImageAsByteArray (url: string) =
        let downloadLink =
            driver.FindElement(By.CssSelector($"a.fa-download[href='https://i.4cdn.org/g/{url}']"))

        downloadLink.Click()
        Threading.Thread.Sleep(4000)
        let path = Directory.EnumerateFiles(appSettings.Selenium.Downloads).First()
        caption <- captionNodePhi3 phi3Model path |> Some
        let bytes = File.ReadAllBytes(path)
        File.Delete(path)
        bytes

    let doStuff () =
        let label, confidence = node.filename.Value |> downloadImageAsByteArray |> identify

        { node with
            caption = caption
            label = Some label
            confidence = Some confidence }

    match
        node.filename.IsSome
        && node.caption.IsNone
        && (node.filename.Value.Contains(".jpg") || node.filename.Value.Contains(".png"))
    with
    | true -> tryWrapper doStuff () |> Result.defaultValue node
    | _ -> node

let caption (driver: FirefoxDriver) (builder: RecapBuilder) =
    let phi3Model = Phi3Model.New()

    driver
        .Navigate()
        .GoToUrl($"https://boards.4chan.org/g/thread/{builder.ThreadId}")

    Threading.Thread.Sleep(4000)

    { builder with
        Chains =
            builder.Chains
            |> Array.map (fun chain ->
                { chain with
                    Nodes = chain.Nodes |> Array.map (fun node -> captionNode driver phi3Model node) }) }

let getIncludedNodes alwaysAddOp rating chain =
    let nodes = new List<ChainNode>()
    let chainmap = new Dictionary<int64, ChainNode>()
    let op = chain.Nodes[0]

    if op.filtered = false && (alwaysAddOp || rating op.rating = true) then
        nodes.Add(op) |> ignore

    for node in chain.Nodes do
        chainmap.Add(node.id, node)

    let rec loop j =
        let replies =
            j.replies
            |> Seq.filter (fun n -> chainmap.ContainsKey(n))
            |> Seq.sortBy (fun n ->
                Seq.max (
                    if chainmap[n].links.Length > 0 then
                        chainmap[n].links
                    else
                        [| 0 |]
                ))

        for r in replies do
            let node = chainmap[r]

            if node.filtered = false && (rating node.rating) then
                nodes.Remove(node) |> ignore
                nodes.Add(node) |> ignore

            loop node

    loop op

    nodes

let minRating r = r >= appSettings.MinimumRating
let minRatingChain r = r >= appSettings.MinimumChainRating
let getIncludedNodesMinRating = getIncludedNodes false (fun r -> minRating r)

let rateChain (recapPluginFunctions: KernelPlugin) chain =
    getIncludedNodes true minRating chain
    |> Seq.map toViewModel
    |> Ok
    |> prettyPrintViewModel
    |> (askDefault recapPluginFunctions["RateChain"])
    |> deserializer.Deserialize<RatingOutput>
    |> fun ratingOutput ->
        { chain with
            Rating = ratingOutput.Rating
            Ratings = Array.concat [| chain.Ratings; [| ratingOutput |] |] }

let rateChainIfUnrated (recapPluginFunctions: KernelPlugin) chain =
    match chain.Rating = -1 with
    | true -> tryWrapper (rateChain recapPluginFunctions) chain |> Result.defaultValue chain
    | false -> chain

let rateChains (recapPluginFunctions: KernelPlugin) (builder: RecapBuilder) =
    match appSettings.RateChain with
    | true ->
        { builder with
            Chains = Array.map (rateChainIfUnrated recapPluginFunctions) builder.Chains }
    | false -> builder

let rateMultipleInChain (recapPluginFunctions: KernelPlugin) (chain: Chain) =
    let chainmap = new Dictionary<int64, ChainNode>()

    for node in chain.Nodes do
        chainmap.Add(node.id, node)

    let setToString by x =
        x |> Seq.map toViewModel |> by |> prettyPrintViewModel

    let unrated = getIncludedNodes false (fun r -> r = -1) chain

    let ratePost () =
        let c =
            getIncludedNodesMinRating chain
            |> setToString (fun f -> Seq.skip (f.Count() - 5) f)

        KernelArguments()
        |> set "chain" c
        |> set "unrated" (unrated |> setToString (Seq.truncate 5))
        |> ask recapPluginFunctions["RateMultiple"]
        |> deserializer.Deserialize<MultiRateOutput[]>
        |> Array.filter (fun r -> chainmap.ContainsKey(r.Id))
        |> Array.iter (fun r ->
            let n = chainmap[r.Id]
            n.rating <- r.Rating

            n.ratings <-
                Array.concat
                    [| chain.Ratings
                       [| { KeyFactors = r.KeyFactors
                            Analysis = r.Analysis
                            Rating = r.Rating } |] |])

    if Seq.isEmpty unrated = false then
        tryWrapper ratePost () |> ignore

    chain

let rateMultiple (recapPluginFunctions: KernelPlugin) (builder: RecapBuilder) =
    match appSettings.RateMultiple with
    | true ->
        { builder with
            Chains = Array.map (rateMultipleInChain recapPluginFunctions) builder.Chains }
    | false -> builder

let categorize (builder: RecapBuilder) =
    let categorizeChain chain =
        match chain.Nodes[0].comment.Contains("https://arxiv.org") with
        | true -> { chain with Category = "Paper" }
        | false -> { chain with Category = "" }

    { builder with
        Chains = Array.map categorizeChain builder.Chains }

let describeChain (recapPluginFunctions: KernelPlugin) (chain: Chain) () =
    getIncludedNodesMinRating chain
    |> Seq.map toViewModel
    |> Seq.truncate appSettings.MaxReplies
    |> prettyPrintViewModel
    |> askDefault recapPluginFunctions["Describe"]
    |> deserializer.Deserialize<DescribeOutput>
    |> fun r -> { chain with Summary = r.Summary }

let describeChainMinRating (recapPluginFunctions: KernelPlugin) (chain: Chain) =
    match minRatingChain chain.Rating with
    | true ->
        match chain.Summary with
        | null
        | "" ->
            tryWrapper (describeChain recapPluginFunctions chain) ()
            |> Result.defaultValue { chain with Summary = "" }
        | _ -> chain
    | false -> chain

let describe (recapPluginFunctions: KernelPlugin) (builder: RecapBuilder) =
    match appSettings.Describe with
    | true ->
        { builder with
            Chains = Array.map (fun c -> describeChainMinRating recapPluginFunctions c) builder.Chains }
    | false -> builder

let recapToText builder =
    let mapChainNodeSeqToString (n: ChainNode seq) =
        n
        |> Seq.map (fun j -> ">" + j.id.ToString())
        |> fun arr -> String.Join(" ", arr).Trim()

    let sb2 = new StringBuilder()

    builder.Chains
    |> Seq.collect (fun c -> c.Nodes)
    |> Seq.filter (fun node ->
        node.label.IsSome
        && (node.label.Value = "miku" || node.label.Value = "teto")
        && node.confidence.Value > 0.85f)
    |> Seq.sortBy (fun c -> c.id)
    |> mapChainNodeSeqToString
    |> sprintf "--Miku (free space):\n%s\n"
    |> sb2.Append
    |> ignore

    builder.Recaps
    |> Seq.map (fun c -> $">>{c.Nodes[0].id}")
    |> String.concat " "
    |> sprintf "\n►Recent Highlight Posts from the Previous Thread: %s"
    |> sb2.Append
    |> ignore

    sb2.Append("\n\nWhy?: 9 reply limit >>102478518\n") |> ignore
    sb2.Append("Fix: https://rentry.org/lmg-recap-script") |> ignore

    let header =
        sprintf "►Recent Highlights from the Previous Thread: >>%s\n\n" builder.ThreadId

    let footer = sb2.ToString()
    let paddingLength = header.Length + footer.Length

    let sb = new StringBuilder()

    let cat2Text c =
        match c with
        | d when d <> "" && d <> "Miku" -> d + ": "
        | _ -> ""

    builder.Chains
    |> Array.filter (fun r -> r.Category = "Paper")
    |> Array.filter (fun r -> r |> getIncludedNodesMinRating |> Seq.length > 1)
    |> Array.iter (fun i ->
        i
        |> getIncludedNodesMinRating
        |> mapChainNodeSeqToString
        |> (fun x -> sprintf "--%s: %s:\n%s\n" (i.Category) i.Summary x)
        |> sb.Append
        |> ignore)

    builder.Chains
    |> Array.filter (fun r -> r.Category = "Paper")
    |> Array.filter (fun r -> r |> getIncludedNodesMinRating |> Seq.length = 1)
    |> Array.map (fun r -> r.Nodes[0])
    |> mapChainNodeSeqToString
    |> fun i ->
        if i.Length = 0 then
            ()
        else
            sprintf "--Papers:\n%s\n" i |> sb.Append |> ignore

    builder.Chains
    |> Array.filter (fun r -> r.Category <> "Paper")
    |> Array.filter (fun r -> minRatingChain r.Rating)
    |> Array.sortByDescending (fun c -> (c.Rating, c.Summary))
    |> Array.iter (fun i ->
        i
        |> getIncludedNodesMinRating
        |> Seq.truncate appSettings.MaxReplies
        |> mapChainNodeSeqToString
        |> (fun x -> sprintf "--%s%s:\n%s\n" (cat2Text i.Category) i.Summary x)
        |> (fun x ->
            if x.Length + sb.Length + paddingLength <= appSettings.MaxLength then
                sb.Append(x) |> ignore
            else
                ()))

    let result = header + sb.ToString() + footer
    sprintf "Recap Length: %i\n" result.Length |> globalLogger.LogInformation
    result

let printRecapHtml builder =
    let strToLink input =
        match String.IsNullOrEmpty(input) with
        | true -> "&nbsp;"
        | false ->
            let pattern = @">(\d{9})"

            let replacement =
                @$"<a href=""https://boards.4chan.org/g/thread/{builder.ThreadId}#p$1"" class=""quotelink"">&gt;&gt;$1</a>"

            let output = Regex.Replace(input, pattern, replacement)
            output

    let sb = new StringBuilder()

    sb
        .Append(
            """<html><head><link rel="stylesheet" title="switch" href="https://s.4cdn.org/css/yotsubluenew.715.css"><link rel="stylesheet" title="switch" href="recap.css"></head><body>"""
        )
        .Append(
            """<div class="boardBanner"><div id="bannerCnt" class="title desktop" data-src="176.jpg" title="Click to change"><img width="300" alt="4chan" src="https://desu-usergeneratedcontent.xyz/g/image/1712/76/1712768281067.png"></div><div class="boardTitle" title="Ctrl/⌘+click to edit board title" spellcheck="false">/lmg/ - Recap</div></div>"""
        )
        .Append(
            """<table id="blotter" class="desktop"><thead><tr><td colspan="2"><hr class="aboveMidAd"></td></tr></thead><tbody id="blotter-msgs">"""
        )
    |> ignore

    (recapToText builder).Replace(">>>", ">>").Split("\n")
    |> Seq.map strToLink
    |> Seq.map (
        sprintf """<tr><td data-utc="1493903967" class="blotter-date"></td><td class="blotter-content">%s</td></tr>"""
    )
    |> Seq.iter (fun rt -> sb.Append(rt) |> ignore)

    sb
        .Append("""<tfoot><tr><td colspan="2"><hr class="aboveMidAd"></td></tr></tfoot>""")
        .Append("</tbody></table>")
    |> ignore

    sb.Append("""<div class="row"><div class="column">""") |> ignore

    let repliesToHtml replies =
        replies
        |> Seq.map (fun r -> sprintf """<a href="#p%i" class="backlink">&gt;&gt;%i</a> """ r r)
        |> String.concat ""

    let chainNodeToHtml j =
        sb
            .Append(
                $"""<div class="postContainer replyContainer noFile" id="pc{j.id}" data-full-i-d="g.{j.id}"><div class="replacedSideArrows" id="sa{j.id}"><a class="hide-reply-button" href="javascript:;"><span class="fa fa-minus-square-o"></span></a></div><div id="p{j.id}" class="post reply"><div class="postInfo desktop" id="pi{j.id}"><input type="checkbox" name="{j.id}" value="delete"> <span class="nameBlock"><span class="name">Anonymous</span> </span>"""
            )
            .Append($""" <span class="dateTime" data-utc="{j.timestamp}">{j.now}</span> """)
            .Append(
                $"""<span class="postNum desktop"><a href="#p{j.id}" title="Link to this post">No.</a><a href="javascript:quote('{j.id}');" title="Reply to this post">{j.id}</a></span>"""
            )
            .Append(
                """<a class="menu-button" href="javascript:;"><i class="fa fa-angle-down"></i></a><span class="container"> """
            )
            .Append(repliesToHtml j.replies)
            .Append("""</span></div>""")
        |> ignore

        if Option.isSome j.filename then
            sb.Append(
                $"""<div class="file" id="f102490301"><div class="fileText" id="fT102490301">
            <a class="fileThumb" href="https://i.4cdn.org/g/{j.filename.Value}" target="_blank"><img src="https://i.4cdn.org/g/{j.filename.Value}" alt="296 KB" data-md5="j6z5wuRT0sPDtl6zHoFnVg==" width=150 loading="lazy"></a>
            </div></div>"""
            )
            |> ignore

        sb
            .Append($"""<blockquote class="postMessage" id="m{j.id}">""")
            .Append(j.unsanitized)
            .Append("</blockquote></div></div>")
        |> ignore

    let max = 5
    let mutable once = true
    let mutable count = 0

    let chainToHtml chain =
        let chainSummary c =
            match c.Category with
            | d when d <> "" && d <> "Miku" -> d + ": " + c.Summary
            | _ -> c.Summary

        count <- count + 1

        if count = max then
            sb.Append("""</div><div class="column">""") |> ignore

            if once = true then
                count <- 1
                once <- false

        sb.Append($"<h2>{chainSummary chain}</h2>") |> ignore
        chain |> getIncludedNodesMinRating |> Seq.iter chainNodeToHtml

    builder.Chains
    |> Array.filter (fun r -> r.Category = "Paper")
    |> Array.filter (fun r -> r |> getIncludedNodesMinRating |> Seq.length > 1)
    |> Array.iter chainToHtml

    builder.Chains
    |> Array.filter (fun r -> r.Category = "Paper")
    |> Array.filter (fun r -> r |> getIncludedNodesMinRating |> Seq.length = 1)
    |> fun arr ->
        (if Array.length arr > 0 then
             sb.Append("<h2>Papers</h2>") |> ignore)

        arr
    |> Array.map (fun r -> r.Nodes[0])
    |> Seq.iter chainNodeToHtml

    builder.Chains
    |> Array.filter (fun r -> r.Category <> "Paper" && minRatingChain r.Rating)
    |> Array.sortByDescending (fun c -> (c.Rating, c.Summary))
    |> Array.iter chainToHtml

    sb.Append("</div></div>") |> ignore

    sb.Append("<h2>Miku (free space)</h2>") |> ignore

    builder.Chains
    |> Seq.collect (fun c -> c.Nodes)
    |> Seq.filter (fun node -> Option.isSome node.filename)
    |> Seq.filter (fun node ->
        node.label.IsSome
        && (node.label.Value = "miku" || node.label.Value = "teto")
        && node.confidence.Value > 0.85f)
    |> Seq.sortBy (fun c -> c.id)
    |> Seq.iter (fun miku ->
        sb.Append($"""<img width=400 src="https://i.4cdn.org/g/{miku.filename.Value}"></img>""")
        |> ignore)

    sb.Append("</body></html>") |> ignore
    File.WriteAllText($"recap-{builder.ThreadId}.html", sb.ToString())
    builder

let saveRecap builder =
    let json = JsonSerializer.Serialize(builder)
    File.WriteAllText($"{builder.ThreadId}.recap.json", json)
    builder

let overrideRating threadNumber postNumber rating =
    threadNumber
    |> createRecapBuilder
    |> loadRecapFromSaveFile
    |> fun i ->
        let updatedChains =
            i.Chains
            |> Array.map (fun chain ->
                if chain.Nodes |> Seq.exists (fun n -> n.id = postNumber) then
                    { chain with Rating = rating }
                else
                    chain)

        { i with Chains = updatedChains }
    |> saveRecap
    |> recapToText
    |> printfn "%s"
    |> ignore

    Task.CompletedTask

let overridePostRating threadNumber postNumber rating =
    threadNumber
    |> createRecapBuilder
    |> loadRecapFromSaveFile
    |> fun i ->
        let updatedChains =
            i.Chains
            |> Array.map (fun chain ->
                let updatedNodes =
                    chain.Nodes
                    |> Array.map (fun node ->
                        if node.id = postNumber then
                            { node with rating = rating }
                        else
                            node)

                { chain with Nodes = updatedNodes })

        { i with Chains = updatedChains }
    |> saveRecap
    |> recapToText
    |> printfn "%s"
    |> ignore

    Task.CompletedTask

let dropSummary threadNumber postNumber =
    threadNumber
    |> createRecapBuilder
    |> loadRecapFromSaveFile
    |> fun i ->
        let updatedChains =
            i.Chains
            |> Array.map (fun chain ->
                if chain.Nodes |> Seq.exists (fun n -> n.id = postNumber) then
                    { chain with Summary = "" }
                else
                    chain)

        { i with Chains = updatedChains }
    |> saveRecap
    |> recapToText
    |> printfn "%s"
    |> ignore

    Task.CompletedTask

let printRecapOnly threadNumber =
    threadNumber
    |> createRecapBuilder
    |> loadRecapFromSaveFile
    |> printRecapHtml
    |> recapToText
    |> printfn "%s"
    |> ignore

let recap threadNumber =
    let driver = new FirefoxDriver(options)
    let timer = new Stopwatch()
    timer.Start()

    threadNumber
    |> createRecapBuilder
    |> loadRecapFromSaveFile
    |> fetchThreadJson driver
    |> saveRecap
    |> caption driver
    |> saveRecap
    |> categorize
    |> rateMultiple recapPluginFunctions
    |> saveRecap
    |> rateChains recapPluginFunctions
    |> saveRecap
    |> describe recapPluginFunctions
    |> saveRecap
    |> recapToText
    |> printfn "%s"

    timer.Stop()
    globalLogger.LogInformation $"Recap completed in {timer.Elapsed}"
    driver.Close()
    driver.Quit()

[<EntryPoint>]
let main argv =
    appSettings <-
        (new ConfigurationBuilder())
            .AddJsonFile("appsettings.json")
            .AddUserSecrets(Assembly.GetExecutingAssembly())
            .AddCommandLine(argv)
            .Build()
            .Get<AppSettings>()

    globalLogger <- (loggerFactory appSettings.Logging.LogLevel.Default).CreateLogger("")

    let argument1 = Argument<string> "threadNumber"
    let argument2 = Argument<string> "filename"
    let argument3 = Argument<int64> "postNumber"
    let argument4 = Argument<int> "rating"

    let command1 =
        CommandLine.Command "recap"
        |> addArgument argument1
        |> setHandler recap argument1

    let command2 =
        CommandLine.Command "load-memories"
        |> addAlias "mem"
        |> addArgument argument2
        |> setHandler importMarkdownFileIntoMemories argument2

    let command3 =
        CommandLine.Command "print-recap"
        |> addArgument argument1
        |> setHandler printRecapOnly argument1

    let command4 =
        CommandLine.Command "override-rating"
        |> addArgument argument1
        |> addArgument argument3
        |> addArgument argument4
        |> setHandler3 overrideRating argument1 argument3 argument4

    let command5 =
        CommandLine.Command "override-post-rating"
        |> addArgument argument1
        |> addArgument argument3
        |> addArgument argument4
        |> setHandler3 overridePostRating argument1 argument3 argument4

    let command6 =
        CommandLine.Command "drop-summary"
        |> addArgument argument1
        |> addArgument argument3
        |> setHandler2 dropSummary argument1 argument3

    RootCommand()
    |> addGlobalOption (CommandLine.Option<int> "--MinimumRating")
    |> addGlobalOption (CommandLine.Option<int> "--MinimumChainRating")
    |> addGlobalOption (CommandLine.Option<int> "--MaxReplies")
    |> addGlobalOption (CommandLine.Option<int> "--MaxLength")
    |> addGlobalOption (CommandLine.Option<Microsoft.Extensions.Logging.LogLevel> "--Logging:LogLevel:Default")
    |> addGlobalOption (CommandLine.Option<bool> "--RateMultiple")
    |> addGlobalOption (CommandLine.Option<bool> "--RateChain")
    |> addGlobalOption (CommandLine.Option<bool> "--Describe")
    |> addCommand command1
    |> addCommand command2
    |> addCommand command3
    |> addCommand command4
    |> addCommand command5
    |> addCommand command6
    |> invoke argv