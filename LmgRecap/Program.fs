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
open Microsoft.SemanticKernel.ChatCompletion
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
open VideoFrameExtractor
open YamlDotNet.Core
open YamlDotNet.Serialization
open YamlDotNet.Serialization.NamingConventions

type CaptionMethod =
    | Disabled = 0
    | Onnx = 1
    | Api = 2

type Website =
    | FourChan = 0
    | GayChan = 1
    | EightChan = 2

type Selenium =
    { Headless: bool
      Downloads: string
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
      Audio: Service
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
      Logging: LoggingConfig
      CaptionMethod: CaptionMethod
      Website: Website }

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

type GayImage =
    { spoiler: bool
      audio: bool
      video: bool
      file_type: int
      thumb_type: int
      length: int
      dims: int array
      size: int
      artist: string
      title: string
      md5: string
      sha1: string
      name: string }

type GayPost =
    { editing: bool
      sage: bool
      auth: int
      id: int64
      time: int64
      body: string
      flag: string
      name: string
      trip: string
      image: GayImage option }

type GayThread = { posts: GayPost[] }

type EightFile =
    { originalName: string
      path: string
      thumb: string
      mime: string
      size: int
      width: int option
      height: int option }

type EightPost =
    { name: string
      signedRole: string option
      email: string option
      id: string option
      subject: string option
      markdown: string
      message: string
      postId: int64
      creation: string
      files: EightFile array }

type EightThread = { posts: EightPost[] }

type CaptionResponse = { content: string }

type Message = { content: string }

type Choice = { message: Message }

type CaptionResponse2 = { choices: Choice[] }

type ChainNodeViewModel =
    { id: int64
      comment: string
      attachment: string }

type ChainViewModel =
    { ReplyChainNumber: int
      Comments: ChainNodeViewModel[] }

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
      attachment = defaultArg node.caption "None." }

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
        .IgnoreUnmatchedProperties()
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

let imageToOnnx (imageBytes: byte array) (size: int) =
    use stream = new MemoryStream(imageBytes)
    use image = Image.Load<Rgb24>(stream)

    let h, w = image.Height, image.Width

    let h', w' =
        if h > w then
            (size, int (float size * float w / float h))
        else
            (int (float size * float h / float w), size)

    image.Mutate(fun c ->
        c.Resize(w', h', KnownResamplers.Lanczos3) |> ignore
        c.Pad(size, size, Color.White) |> ignore)

    let width = image.Width
    let height = image.Height

    for y in 0 .. (height - h') / 2 do
        for x in 0 .. width - 1 do
            image[x, y] <- image[x, (height - h') / 2 + 1]

    for y in height - (height - h') / 2 .. height - 1 do
        for x in 0 .. width - 1 do
            image[x, y] <- image[x, height - (height - h') / 2 - 1]

    for y in 0 .. height - 1 do
        for x in 0 .. (width - w') / 2 do
            image[x, y] <- image[(width - w') / 2 + 1, y]

    for y in 0 .. height - 1 do
        for x in width - (width - w') / 2 .. width - 1 do
            image[x, y] <- image[width - (width - w') / 2 - 1, y]

    let tensor = new DenseTensor<float32>([| 1; size; size; 3 |])

    for y = 0 to size - 1 do
        for x = 0 to size - 1 do
            let pixel = image[x, y]
            tensor[0, y, x, 0] <- float32 pixel.R / 255.0f
            tensor[0, y, x, 1] <- float32 pixel.G / 255.0f
            tensor[0, y, x, 2] <- float32 pixel.B / 255.0f

    tensor

let identify (session: InferenceSession) (imageBytes: byte array) =
    let tensor = imageToOnnx imageBytes 512
    let inputs = [ NamedOnnxValue.CreateFromTensor("inputs", tensor) ]
    let results = session.Run(inputs)
    let outputTensor = results[0].AsTensor<float32>()
    let probs = outputTensor.ToArray()

    let allowed =
        [| "kasane_teto"; "hatsune_miku"; "kagamine_rin"; "akita_neru"; "yowane_haku"; "megurine_luka" |]

    let tags =
        session.ModelMetadata.CustomMetadataMap["tags"]
        |> JsonSerializer.Deserialize<string[]>

    Array.zip tags probs
    |> Array.filter (fun (_, score) -> score >= 0.5f)
    |> Array.filter (fun (tag, _) -> allowed.Contains(tag))
    |> Array.sortByDescending snd
    |> Array.tryHead
    |> Option.defaultValue ("other", 1f)

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
    let aiClient service =
        let clientOptions = new OpenAIClientOptions()
        clientOptions.Endpoint <- new Uri(service.Endpoint)
        clientOptions.NetworkTimeout <- new TimeSpan(2, 0, 0)

        new OpenAIClient(new ClientModel.ApiKeyCredential(service.Key), clientOptions)

    let builder = Kernel.CreateBuilder()

    builder.Services.AddSingleton((loggerFactory appSettings.Logging.LogLevel.Default))
    |> ignore

    builder
        .AddOpenAIChatCompletion(appSettings.Completion.Model, aiClient appSettings.Completion, "Completion")
        .AddOpenAIChatCompletion(appSettings.Multimodal.Model, aiClient appSettings.Multimodal, "Multimodal")
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
    |> fun s ->
        let idx = s.IndexOf("</think>\n\n")

        if idx = -1 then
            s
        else
            s.Substring(idx + "</think>\n\n".Length)
    |> fun i ->
        globalLogger.LogDebug("Response:\n" + i)
        i

let askDefault kernelFunction prompt =
    let openAIPromptExecutionSettings = new OpenAIPromptExecutionSettings()
    openAIPromptExecutionSettings.ServiceId <- "Completion"

    KernelArguments(openAIPromptExecutionSettings)
    |> set "input" prompt
    |> ask kernelFunction

let loadRecapFromSaveFile builder =
    let filename = $"{builder.ThreadId}.recap.json"

    match File.Exists(filename) with
    | false -> builder
    | true -> filename |> File.ReadAllText |> JsonSerializer.Deserialize<RecapBuilder>

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

let mapEightPostToChainNode post =
    let getUnixTimestamp (datetimeString: string) : int64 =
        let format = "yyyy-MM-ddTHH:mm:ss.fffZ"

        match
            DateTime.TryParseExact(datetimeString, format, null, System.Globalization.DateTimeStyles.AssumeUniversal)
        with
        | (true, datetimeObject) ->
            let epoch = DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)
            int64 (datetimeObject - epoch).TotalSeconds
        | (false, _) -> failwith "Invalid datetime format"

    let ts = getUnixTimestamp post.creation

    { id = post.postId
      timestamp = ts
      now =
        DateTimeOffset
            .FromUnixTimeSeconds(ts)
            .UtcDateTime.ToString("dd MMM yyyy (ddd) HH:mm:ss")
      filtered = appSettings.Filters.Any(fun filter -> Regex.IsMatch(post.markdown.ToLower(), filter))
      links = [||]
      replies = [||]
      ratings = [||]
      comment = ""
      unsanitized = post.markdown
      context = None
      rating = -1
      reasoning = ""
      filename =
        if post.files.Length > 0 then
            Some post.files[0].originalName
        else
            None
      caption = None
      label = None
      confidence = None }

let mapGayPostToChainNode (post: GayPost) =
    let getFileExtension (fileType: int) =
        match fileType with
        | 0 -> ".jpg"
        | 1 -> ".png"
        | 2 -> ".gif"
        | 3 -> ".webm"
        | 6 -> ".mp4"
        | _ -> ""

    { id = post.id
      timestamp = post.time
      now =
        DateTimeOffset
            .FromUnixTimeSeconds(post.time)
            .UtcDateTime.ToString("dd MMM yyyy (ddd) HH:mm:ss")
      filtered = appSettings.Filters.Any(fun filter -> Regex.IsMatch(post.body.ToLower(), filter))
      links = [||]
      replies = [||]
      ratings = [||]
      comment = ""
      unsanitized = post.body
      context = None
      rating = -1
      reasoning = ""
      filename =
        match post.image with
        | Some image -> Some(image.sha1 + getFileExtension image.file_type)
        | None -> None
      caption = None
      label = None
      confidence = None }

let buildReferences (chainmap: Dictionary<int64, ChainNode>) node =
    let quoteIndicator =
        match appSettings.Website with
        | Website.EightChan
        | Website.FourChan -> "&gt;&gt;"
        | Website.GayChan -> @"\u003E\u003E"
        | _ -> failwith "Invalid Website Selection"

    { node with
        links =
            Regex.Matches(node.unsanitized, quoteIndicator + @"(\d{" + node.id.ToString().Length.ToString() + "})")
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
                (match appSettings.Website with
                 | Website.EightChan
                 | Website.FourChan ->
                     node.unsanitized.Replace("<br>", "\n")
                     |> stripTags
                     |> HttpUtility.HtmlDecode
                     |> fun c -> c.Replace("https://arxiv.org/pdf", "https://arxiv.org/abs").Trim()
                     |> fun c ->
                         (Regex.Matches(
                             c,
                             @"https:\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?"
                          )
                          |> Seq.map (fun m ->
                              m.Value,
                              tryWrapper (fetchTitleFromUrlAsync m.Value) ()
                              |> Result.defaultValue "Error fetching title")
                          |> Seq.fold
                              (fun (newText: string) (url, title) -> newText.Replace(url, $"[{title}]({url})"))
                              c)
                 | Website.GayChan ->
                     node.unsanitized
                     |> HttpUtility.HtmlDecode
                     |> fun c -> c.Replace("https://arxiv.org/pdf", "https://arxiv.org/abs").Trim()
                     |> fun c ->
                         (Regex.Matches(
                             c,
                             @"https:\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?"
                          )
                          |> Seq.map (fun m ->
                              m.Value,
                              tryWrapper (fetchTitleFromUrlAsync m.Value) ()
                              |> Result.defaultValue "Error fetching title")
                          |> Seq.fold
                              (fun (newText: string) (url, title) -> newText.Replace(url, $"[{title}]({url})"))
                              c)
                 | _ -> failwith "Invalid Website Selection") }
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

    let updateReferences oldbuilder =
        { oldbuilder with
            Chains =
                oldbuilder.Chains
                |> Array.map (fun chain ->
                    { chain with
                        Nodes = chain.Nodes |> Array.map (buildReferences cnm) }) }

    (match appSettings.Website with
     | Website.FourChan ->
         driver
             .Navigate()
             .GoToUrl($"https://a.4cdn.org/g/thread/{builder.ThreadId}.json")

         driver.FindElement(By.TagName("html")).Text
         |> JsonSerializer.Deserialize<Thread>
         |> fun a -> a.posts[1..]
         |> Array.filter (fun p -> p.no > maxId)
         |> Array.map mapPostToChainNode
     | Website.GayChan ->
         driver.Navigate().GoToUrl($"https://meta.4chan.gay/tech/{builder.ThreadId}")

         driver.FindElement(By.Id("post-data")).GetAttribute("textContent")
         |> JsonSerializer.Deserialize<GayThread>
         |> fun a -> a.posts
         |> Array.filter (fun p -> p.id > maxId)
         |> Array.map mapGayPostToChainNode
     | Website.EightChan ->
         driver.Navigate().GoToUrl($"https://8chan.moe/ais/res/{builder.ThreadId}.json")
         Thread.Sleep(2000)
         driver.FindElement(By.CssSelector("h1 > a")).Click()

         driver.FindElement(By.TagName("pre")).Text
         |> JsonSerializer.Deserialize<EightThread>
         |> fun a -> a.posts
         |> Array.filter (fun p -> p.postId > maxId)
         |> Array.map mapEightPostToChainNode
     | _ -> failwith "Invalid Website Selection")
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

    static member Empty() =
        { Path = ""
          SystemPrompt = ""
          UserPrompt = ""
          FullPrompt = ""
          Model = null
          Processor = null
          TokenizerStream = null }

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
    Some(response.ToString().Trim())

let getMimeType (file: string) =
    let extension = Path.GetExtension(file).ToLower()

    match extension with
    | ".jpg"
    | ".jpeg" -> "image/jpeg"
    | ".png" -> "image/png"
    | ".gif" -> "image/gif"
    | _ -> "image/jpeg"

let captionNodeApi (file: string) =
    let chat =
        kernel.Services.GetRequiredKeyedService<IChatCompletionService>("Multimodal")

    let history = new ChatHistory()
    history.AddSystemMessage("You are an image captioner that responds to questions directly.")

    let message = new ChatMessageContentItemCollection()
    message.Add(new TextContent("Describe what is in the image."))

    let bytes = File.ReadAllBytes(file)
    let mimeType = getMimeType file
    message.Add(new ImageContent(bytes, mimeType))
    history.AddUserMessage(message)

    let result = chat.GetChatMessageContentAsync(history).Result
    globalLogger.LogInformation("Generation complete: {GeneratorResponse}", result.Content)

    let s = result.Content
    let idx = s.IndexOf("<answer>")

    if idx = -1 then
        Some s
    else
        Some (s.Substring(idx + "<answer>".Length))

let downloadImage (driver: FirefoxDriver) (url: string) =
    let downloadLink =
        match appSettings.Website with
        | Website.FourChan -> driver.FindElement(By.CssSelector($"a[href='{url}'][download]"))
        | Website.GayChan -> driver.FindElement(By.CssSelector($"a[href='{url}'][download]"))
        | Website.EightChan -> driver.FindElement(By.CssSelector($"a[href][download='{url}']"))
        | _ -> failwith "Invalid Website Selection"

    downloadLink.SendKeys(Keys.Return)

    let rec waitForDownload (filePaths: IEnumerable<string>) =
        if filePaths.Count() = 1 && filePaths.First().EndsWith(".part") <> true then
            filePaths.First()
        else
            Threading.Thread.Sleep(1000)
            Directory.EnumerateFiles(appSettings.Selenium.Downloads) |> waitForDownload

    Directory.EnumerateFiles(appSettings.Selenium.Downloads) |> waitForDownload

let identifyNode session (path: string) node =
    let bytes = File.ReadAllBytes(path)
    let label, confidence = identify session bytes

    { node with
        label = Some label
        confidence = Some confidence }

let tryCaptionIdentify (driver: FirefoxDriver) (session: InferenceSession) (phi3Model) (node) =
    let url =
        match appSettings.Website with
        | Website.FourChan -> $"https://i.4cdn.org/g/{node.filename.Value}"
        | Website.GayChan -> $"/assets/images/src/{node.filename.Value}"
        | Website.EightChan -> node.filename.Value
        | _ -> failwith "Invalid Website Selection"

    let downloadedPath = downloadImage driver url

    let processedPath =
        if
            downloadedPath.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase)
            || downloadedPath.EndsWith(".webm", StringComparison.OrdinalIgnoreCase)
        then
            let bytes = getMiddleFrameBytes downloadedPath
            let tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".jpg")
            File.WriteAllBytes(tempPath, bytes)
            tempPath
        else
            downloadedPath

    try
        match appSettings.CaptionMethod with
        | CaptionMethod.Onnx -> captionNodePhi3 phi3Model processedPath
        | CaptionMethod.Api -> captionNodeApi processedPath
        | _ -> node.caption
        |> fun caption -> { node with caption = caption }
        |> identifyNode session processedPath
    finally
        File.Delete(downloadedPath)

        if processedPath <> downloadedPath then
            File.Delete(processedPath)

let captionNode (driver: FirefoxDriver) session phi3Model node =
    Directory.EnumerateFiles(appSettings.Selenium.Downloads)
    |> Seq.iter (fun f -> File.Delete(f))

    match node.filename, node.caption with
    | Some _, None ->
        tryWrapper (tryCaptionIdentify driver session phi3Model) node
        |> Result.defaultValue node
    | _ -> node

let caption (driver: FirefoxDriver) (builder: RecapBuilder) =
    let sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0)
    use session = new InferenceSession(appSettings.ResnetModelPath, sessionOptions)

    let phi3Model =
        match appSettings.CaptionMethod with
        | CaptionMethod.Onnx -> Phi3Model.New()
        | _ -> Phi3Model.Empty()

    match appSettings.Website with
    | Website.FourChan ->
        driver
            .Navigate()
            .GoToUrl($"https://boards.4chan.org/g/thread/{builder.ThreadId}")
    | Website.GayChan -> driver.Navigate().GoToUrl($"https://meta.4chan.gay/tech/{builder.ThreadId}")
    | Website.EightChan -> driver.Navigate().GoToUrl($"https://8chan.moe/ais/res/{builder.ThreadId}.html")
    | _ -> failwith "Invalid Website Selection"

    Threading.Thread.Sleep(4000)

    { builder with
        Chains =
            builder.Chains
            |> Array.map (fun chain ->
                { chain with
                    Nodes = chain.Nodes |> Array.map (captionNode driver session phi3Model) }) }

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
            |> Seq.filter (fun n -> n > j.id)
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
            |> setToString (fun f -> Seq.skip (f.Count() - 15) f)

        let openAIPromptExecutionSettings = new OpenAIPromptExecutionSettings()
        openAIPromptExecutionSettings.ServiceId <- "Completion"

        KernelArguments(openAIPromptExecutionSettings)
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
        match
            chain.Nodes[0].comment.Contains("https://arxiv.org")
            || chain.Nodes[0].comment.Contains("https://huggingface.co/papers")
        with
        | true -> { chain with Category = "Paper" }
        | false -> chain

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

let recapToYaml builder =
    builder.Chains
    |> Array.filter (fun r -> r.Rating >= 3)
    |> Array.filter (fun r -> (getIncludedNodes false (fun c -> c >= 3) r) |> Seq.length > 0)
    |> Array.mapi (fun i r ->
        { ReplyChainNumber = i
          Comments = Array.map toViewModel r.Nodes })
    |> prettyPrintViewModel

let recapToYamlFile builder =
    recapToYaml builder
    |> fun s -> File.WriteAllText($"recap-{builder.ThreadId}.yaml", s)

    builder

let recapToText builder =
    let mapChainNodeSeqToString (n: ChainNode seq) =
        n
        |> Seq.map (fun j -> ">" + j.id.ToString())
        |> fun arr -> String.Join(" ", arr).Trim()

    let sb2 = new StringBuilder()

    builder.Chains
    |> Seq.collect (fun c -> c.Nodes)
    |> Seq.filter (fun node -> node.label.IsSome && (node.label.Value <> "other"))
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
        |> Seq.truncate appSettings.MaxReplies
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
    |> Array.sortByDescending (fun c -> (c.Rating, (c.Nodes.Length), c.Summary))
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

    builder.Chains
    |> Seq.filter (fun chain -> chain.Rating = -1)
    |> Seq.length
    |> fun count -> sprintf "Remaining unrated chains: %i\n" count
    |> globalLogger.LogInformation

    builder.Chains
    |> Seq.collect (fun c -> c.Nodes)
    |> Seq.filter (fun node -> node.rating = -1 && node.filtered = false)
    |> Seq.length
    |> fun count -> sprintf "Remaining unrated nodes: %i\n" count
    |> globalLogger.LogInformation

    let result = header + sb.ToString() + footer
    sprintf "Recap Length: %i\n" result.Length |> globalLogger.LogInformation
    result

let printRecapHtml builder =
    let strToLink input =
        match String.IsNullOrEmpty(input) with
        | true -> "&nbsp;"
        | false ->
            let pattern = @">(\d{5,9})"

            let replacement = @$"<a href=""#p$1"" class=""quotelink"">&gt;&gt;$1</a>"

            let input2 = input.Replace(">>", ">")
            Regex.Replace(input2, pattern, replacement)

    let sb = new StringBuilder()

    sb
        .Append(
            """<html><head><link rel="stylesheet" title="switch" href="yotsubluenew.css"><link rel="stylesheet" title="switch" href="recap.css"></head><body>"""
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

    sb.Append("""<div class="row"><div>""") |> ignore

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

        let mardownLinksToHtml a =
            Regex.Replace(a, @"\[(.*?)\]\((.*?)\)", "<a href=\"$2\">$2</a>")

        sb
            .Append($"""<blockquote class="postMessage" id="m{j.id}" style="white-space: pre-wrap;">""")
            .Append(strToLink (mardownLinksToHtml j.comment))
            .Append("</blockquote></div></div>")
        |> ignore

    let chainToHtml chain =
        let chainSummary c =
            match c.Category with
            | d when d <> "" && d <> "Miku" -> d + ": " + c.Summary
            | _ -> c.Summary

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
    |> Array.sortByDescending (fun c -> (c.Rating, (c.Nodes.Length), c.Summary))
    |> Array.iter chainToHtml

    sb.Append("</div></div>") |> ignore

    sb.Append("<h2>Miku (free space)</h2>") |> ignore

    builder.Chains
    |> Seq.collect (fun c -> c.Nodes)
    |> Seq.filter (fun node -> Option.isSome node.filename)
    |> Seq.filter (fun node -> node.label.IsSome && (node.label.Value <> "other"))
    |> Seq.sortBy (fun c -> c.id)
    |> Seq.iter (fun miku ->
        sb.Append($"""<a id="p{miku.id}"><img width=400 src="https://i.4cdn.org/g/{miku.filename.Value}"></img></a>""")
        |> ignore)

    sb.Append("</body></html>") |> ignore
    File.WriteAllText($"recap-{builder.ThreadId}.html", sb.ToString())
    builder

let generateHtmlTable (file1Path: string) (file2Path: string) =
    let loadAndPrepareNodes (filePath: string) =
        filePath
        |> File.ReadAllText
        |> JsonSerializer.Deserialize<RecapBuilder>
        |> fun recapBuilder ->
            recapBuilder.Chains
            |> Array.collect (fun c -> c.Nodes)
            |> Array.append (recapBuilder.Recaps |> Array.collect (fun c -> c.Nodes))
            |> Array.filter (fun n -> n.filename.IsSome && n.caption.IsSome)
            |> Array.sortBy (fun n -> n.filename.Value)

    let nodes1 = loadAndPrepareNodes file1Path
    let nodes2 = loadAndPrepareNodes file2Path

    let pairedNodes =
        nodes1
        |> Array.zip nodes2
        |> Array.choose (fun (n1, n2) ->
            if n1.filename = n2.filename then
                Some(n1.caption.Value, n1.filename.Value, n2.caption.Value)
            else
                None)

    let html =
        let mediaTag (f: string) =
            let ext = Path.GetExtension(f).ToLower()

            match ext with
            | ".mp4"
            | ".webm" ->
                $"""<video><source src="https://i.4cdn.org/g/{f}" type="video/{ext.Replace(".", "")}"></video>"""
            | _ -> $"<img src=\"https://i.4cdn.org/g/{f}\">"

        let tableRow (c1, f, c2) =
            $"<tr><td class=\"text-col\"><pre>{c1}</pre></td><td class=\"image-col\">{mediaTag f}</td><td class=\"text-col\"><pre>{c2}</pre></td></tr>"

        let tableBody = pairedNodes |> Array.map tableRow |> String.concat ""

        $"<html>
            <head>
                <style>
                    table {{
                        width: 100%%;
                        table-layout: fixed;
                    }}
                    td {{
                        padding: 10px;
                        vertical-align: top;
                    }}
                    .text-col {{
                        width: 40%%;
                        word-wrap: break-word;
                    }}
                    .image-col {{
                        width: 20%%;
                    }}
                    img, video {{
                        max-width: 100%%;
                        height: auto;
                        max-height: 250px;
                        display: block;
                        margin: 0 auto;
                    }}
                    pre {{
                        white-space: pre-wrap;
                    }}
                </style>
            </head>
            <body>
                <table border='1'>
                    <tr><th>First Caption</th><th>Image</th><th>Second Caption</th></tr>
                    {tableBody}
                </table>
            </body>
        </html>"

    File.WriteAllText("output.html", html)

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

let threadSummaryRecap threadNumber =
    threadNumber
    |> createRecapBuilder
    |> loadRecapFromSaveFile
    |> recapToYaml
    |> (askDefault recapPluginFunctions["ThreadSummary"])
    |> printfn "%s"
    |> ignore

let generateSpeech (outputPath: string) (text: string) =
    if File.Exists(outputPath) then
        printfn $"Audio file already exists at {outputPath}"
    else
        try
            let clientOptions = new OpenAIClientOptions()
            clientOptions.Endpoint <- new Uri(appSettings.Audio.Endpoint)
            clientOptions.NetworkTimeout <- new TimeSpan(2, 0, 0)

            let client =
                new OpenAIClient(new ClientModel.ApiKeyCredential(appSettings.Audio.Key), clientOptions)

            let audioClient = client.GetAudioClient("kokoro")
            let response = audioClient.GenerateSpeechAsync(text, "af_miku").Result.Value
            use stream = File.Create(outputPath)
            response.ToStream().CopyTo(stream)
            globalLogger.LogInformation $"Audio segment saved to {outputPath}"
        with e ->
            globalLogger.LogError $"Error creating Audio segment {outputPath}: {e.Message}"

    text

let generateNewscasterScript threadNumber =
    threadNumber
    |> createRecapBuilder
    |> loadRecapFromSaveFile
    |> recapToText
    |> (fun s -> s.Split('\n'))
    |> Array.filter (fun line -> line.StartsWith "--")
    |> Array.map (fun line -> line.Replace("--", "").Replace(": ", ""))
    |> String.concat "\n"
    |> (askDefault recapPluginFunctions["NewscasterScript"])
    |> generateSpeech "output_audio.wav"
    |> printfn "%s"
    |> ignore

let printRecapOnly threadNumber =
    threadNumber
    |> createRecapBuilder
    |> loadRecapFromSaveFile
    |> printRecapHtml
    |> recapToYamlFile
    |> recapToText
    |> printfn "%s"
    |> ignore

let buildWebDriver () =
    let options = new FirefoxOptions()

    if appSettings.Selenium.Headless then
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
    new FirefoxDriver(options)

let recap threadNumber =
    let driver = buildWebDriver ()
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

let monitorThread threadNumber =
    let driver = buildWebDriver ()

    try
        while true do
            let builder =
                threadNumber
                |> createRecapBuilder
                |> loadRecapFromSaveFile
                |> fetchThreadJson driver
                |> saveRecap
                |> caption driver
                |> saveRecap
                |> categorize

            let totalPosts = builder.Chains |> Seq.collect (fun c -> c.Nodes) |> Seq.length

            let unratedChains =
                builder.Chains |> Seq.filter (fun c -> c.Rating = -1) |> Seq.length

            let unratedPosts =
                builder.Chains
                |> Seq.collect (fun c -> c.Nodes)
                |> Seq.filter (fun node -> node.rating = -1 && node.filtered = false)
                |> Seq.length

            globalLogger.LogInformation
                $"Monitoring thread {threadNumber} - Total Posts: {totalPosts}, Unrated Posts: {unratedPosts}, Unrated Chains: {unratedChains}"

            builder
            |> (if unratedPosts > 20 then
                    rateMultiple recapPluginFunctions
                else
                    id)
            |> (if unratedChains > 30 && unratedPosts < 50 then
                    rateChains recapPluginFunctions
                else
                    id)
            |> (if totalPosts > 300 then
                    describe recapPluginFunctions
                else
                    id)
            |> saveRecap
            |> recapToText
            |> printfn "%s"

            if Environment.OSVersion.Platform = PlatformID.Win32NT then
                Console.Beep()
            else
                printf "\a"

            Thread.Sleep(1800000)
    finally
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
    let argument5 = Argument<string> "filename1"
    let argument6 = Argument<string> "filename2"

    let command1 =
        CommandLine.Command "recap"
        |> addArgument argument1
        |> setHandler1 recap argument1

    let command2 =
        CommandLine.Command "load-memories"
        |> addAlias "mem"
        |> addArgument argument2
        |> setHandler1 importMarkdownFileIntoMemories argument2

    let command3 =
        CommandLine.Command "print-recap"
        |> addArgument argument1
        |> setHandler1 printRecapOnly argument1

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

    let command7 =
        CommandLine.Command "thread-summary"
        |> addArgument argument1
        |> setHandler1 threadSummaryRecap argument1

    let command8 =
        CommandLine.Command "generate-newscaster-script"
        |> addArgument argument1
        |> setHandler1 generateNewscasterScript argument1

    let command9 =
        CommandLine.Command "monitor"
        |> addArgument argument1
        |> setHandler1 monitorThread argument1

    let command10 =
        CommandLine.Command "table"
        |> addArgument argument5
        |> addArgument argument6
        |> setHandler2 generateHtmlTable argument5 argument6

    RootCommand()
    |> addGlobalOption (CommandLine.Option<int> "--MinimumRating")
    |> addGlobalOption (CommandLine.Option<int> "--MinimumChainRating")
    |> addGlobalOption (CommandLine.Option<int> "--MaxReplies")
    |> addGlobalOption (CommandLine.Option<int> "--MaxLength")
    |> addGlobalOption (CommandLine.Option<bool> "--Selenium:Headless")
    |> addGlobalOption (CommandLine.Option<Microsoft.Extensions.Logging.LogLevel> "--Logging:LogLevel:Default")
    |> addGlobalOption (CommandLine.Option<bool> "--RateMultiple")
    |> addGlobalOption (CommandLine.Option<bool> "--RateChain")
    |> addGlobalOption (CommandLine.Option<bool> "--Describe")
    |> addGlobalOption (CommandLine.Option<CaptionMethod> "--CaptionMethod")
    |> addGlobalOption (CommandLine.Option<Website> "--Website")
    |> addCommand command1
    |> addCommand command2
    |> addCommand command3
    |> addCommand command4
    |> addCommand command5
    |> addCommand command6
    |> addCommand command7
    |> addCommand command8
    |> addCommand command9
    |> addCommand command10
    |> invoke argv
