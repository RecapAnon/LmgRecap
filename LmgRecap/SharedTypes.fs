namespace LmgRecap.Shared

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
      Website: Website
      AllowedFreeSpaceTags: string[] }

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

[<CLIMutable>]
type SortChainViewModel =
    { Id: int64
      Summary: string
      Analysis: string
      ReplyCount: int
      SortIndex: int }

[<CLIMutable>]
type SortOutput = { Id: int64; SortIndex: int }

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
      Summary: string
      SortIndex: int }

type RecapBuilder =
    { ThreadId: string
      Chains: Chain[]
      Recaps: Chain[] }
