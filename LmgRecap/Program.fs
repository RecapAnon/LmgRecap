module LmgRecap.Program

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
open CommandLineExtensions
open Logging
open Microsoft.Extensions.Configuration
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open Microsoft.SemanticKernel
open Microsoft.SemanticKernel.ChatCompletion
open Microsoft.SemanticKernel.Connectors.OpenAI
open OpenAI
open OpenQA.Selenium
open OpenQA.Selenium.Firefox
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
    | Meguca = 1
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
      UseCuda: bool
      WDModelPath: string
      WDLabelPath: string
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

type MegucaImage =
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

type MegucaPost =
    { editing: bool
      sage: bool
      auth: int
      id: int64
      time: int64
      body: string
      flag: string
      name: string
      trip: string
      image: MegucaImage option }

type MegucaThread = { posts: MegucaPost[] }

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

let set i v (a: KernelArguments) =
    a[i] <- v
    a

let ask kernelFunction (kernelArguments: KernelArguments) =
    for key in kernelArguments.Names do
        globalLogger.LogDebug(key + ":\n" + kernelArguments[key].ToString())

    kernelArguments
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

let mapMegucaPostToChainNode (post: MegucaPost) =
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
        | Website.Meguca -> @"\u003E\u003E"
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
                 | Website.Meguca ->
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
     | Website.Meguca ->
         driver.Navigate().GoToUrl($"https://meta.4chan.gay/tech/{builder.ThreadId}")

         driver.FindElement(By.Id("post-data")).GetAttribute("textContent")
         |> JsonSerializer.Deserialize<MegucaThread>
         |> fun a -> a.posts
         |> Array.filter (fun p -> p.id > maxId)
         |> Array.map mapMegucaPostToChainNode
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

    let message = new ChatMessageContentItemCollection()
    message.Add(new TextContent("Describe the image."))

    let bytes = File.ReadAllBytes(file)
    let mimeType = getMimeType file
    message.Add(new ImageContent(bytes, mimeType))
    history.AddUserMessage(message)

    let result = chat.GetChatMessageContentAsync(history).Result
    globalLogger.LogInformation("Generation complete: {GeneratorResponse}", result.Content)

    history.AddAssistantMessage(result.Content)

    history.AddUserMessage(
        "If the image is a screenshot of a chat with an AI assistant, respond with 'Chatlog'. Otherwise, respond with 'Image'"
    )

    let resultCategory = chat.GetChatMessageContentAsync(history).Result
    globalLogger.LogInformation("Generation complete: {GeneratorResponse}", resultCategory.Content)
    Some(resultCategory.Content + ". " + result.Content)

let downloadImage (driver: FirefoxDriver) (url: string) =
    let downloadLink =
        match appSettings.Website with
        | Website.FourChan -> driver.FindElement(By.CssSelector($"a[href='{url}'][download]"))
        | Website.Meguca -> driver.FindElement(By.CssSelector($"a[href='{url}'][download]"))
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

let getWaifuTags (logger: ILogger) (tagger: WaifuDiffusionPredictor option) (bytes: byte array) : string =
    match tagger with
    | Some t ->
        let allowed =
            [| "kasane teto"
               "hatsune miku"
               "kagamine rin"
               "akita neru"
               "yowane haku"
               "megurine luka" |]

        let result = t.predict bytes 0.35 true 0.85 true
        let ratingTags = result.RatingTags |> Array.map fst
        let generalTags = result.GeneralTags |> Array.map fst
        let characterTags = result.CharacterTags |> Array.map fst
        let allTags = Array.concat [ ratingTags; generalTags; characterTags ]
        logger.LogInformation("WaifuDiffusion Tags: {Tags}", allTags)

        result.CharacterTags
        |> Array.map (fun (tag, score) -> ((tag.Replace("character:", "")), score))
        |> Array.filter (fun (tag, _) -> allowed.Contains(tag))
        |> Array.sortByDescending snd
        |> Array.tryHead
        |> Option.map fst
        |> Option.defaultValue "other"
    | None -> "other"

let identifyNode tagger (path: string) node =
    let bytes = File.ReadAllBytes(path)
    let label = getWaifuTags globalLogger tagger bytes

    { node with
        label = Some label
        confidence = Some 1.0f }

let tryCaptionIdentify (driver: FirefoxDriver) (tagger: WaifuDiffusionPredictor option) (node) =
    let url =
        match appSettings.Website with
        | Website.FourChan -> $"https://i.4cdn.org/g/{node.filename.Value}"
        | Website.Meguca -> $"/assets/images/src/{node.filename.Value}"
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
        | CaptionMethod.Onnx -> None
        | CaptionMethod.Api -> captionNodeApi processedPath
        | _ -> node.caption
        |> fun caption -> { node with caption = caption }
        |> identifyNode tagger processedPath
    finally
        File.Delete(downloadedPath)

        if processedPath <> downloadedPath then
            File.Delete(processedPath)

let captionNode (driver: FirefoxDriver) session node =
    Directory.EnumerateFiles(appSettings.Selenium.Downloads)
    |> Seq.iter (fun f -> File.Delete(f))

    match node.filename, node.caption with
    | Some _, None -> tryWrapper (tryCaptionIdentify driver session) node |> Result.defaultValue node
    | _ -> node

let caption (driver: FirefoxDriver) (builder: RecapBuilder) =
    let tagger =
        try
            match Option.ofObj appSettings.WDModelPath, Option.ofObj appSettings.WDLabelPath with
            | Some modelPath, Some labelPath ->
                Some(new WaifuDiffusionPredictor(modelPath, labelPath, appSettings.UseCuda))
            | _ ->
                logger.LogWarning("Failed to initialize WaifuDiffusionPredictor: Missing configuration.")
                None
        with ex ->
            logger.LogError(ex, "Failed to initialize WaifuDiffusionPredictor: {Error}", ex.Message)
            None

    match appSettings.Website with
    | Website.FourChan ->
        driver
            .Navigate()
            .GoToUrl($"https://boards.4chan.org/g/thread/{builder.ThreadId}")
    | Website.Meguca -> driver.Navigate().GoToUrl($"https://meta.4chan.gay/tech/{builder.ThreadId}")
    | Website.EightChan -> driver.Navigate().GoToUrl($"https://8chan.moe/ais/res/{builder.ThreadId}.html")
    | _ -> failwith "Invalid Website Selection"

    Threading.Thread.Sleep(4000)

    { builder with
        Chains =
            builder.Chains
            |> Array.map (fun chain ->
                { chain with
                    Nodes = chain.Nodes |> Array.map (captionNode driver tagger) }) }

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
        |> set "unrated" (unrated |> setToString (Seq.truncate 15))
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
            || chain.Nodes[0].comment.Contains("https://www.arxiv.org")
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

    sb2.Append("\n\nWhy?: >>102478518\n") |> ignore
    sb2.Append("Enable Links: https://rentry.org/lmg-recap-script") |> ignore

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
    |> addCommand command3
    |> addCommand command4
    |> addCommand command5
    |> addCommand command6
    |> addCommand command7
    |> addCommand command8
    |> addCommand command9
    |> invoke argv
