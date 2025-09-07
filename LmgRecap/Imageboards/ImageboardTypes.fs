namespace LmgRecap.Imageboards

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

type PostType =
    | FourChanPost of Post
    | EightChanPost of EightPost
    | MegucaPost of MegucaPost

open System.Collections.Generic
open OpenQA.Selenium.Firefox
open LmgRecap.Shared
open OpenQA.Selenium

type IImageboardHandler =
    abstract member FetchThreadJson:
        driver: FirefoxDriver * builder: RecapBuilder * maxId: int64 * filters: string array -> ChainNode array

    abstract member MapPostToChainNode: filters: string array * post: PostType -> ChainNode
    abstract member BuildReferences: chainmap: Dictionary<int64, ChainNode> * node: ChainNode -> ChainNode
    abstract member Sanitize: node: ChainNode * fetchTitleFromUrlAsync: (string -> Result<string, string>) -> ChainNode
    abstract member QuoteIndicator: string
    abstract member GetDownloadLink: driver: FirefoxDriver * url: string -> IWebElement
    abstract member GetImageUrl: filename: string -> string
    abstract member GetThreadUrl: threadId: string -> string
