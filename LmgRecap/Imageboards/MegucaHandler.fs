namespace LmgRecap.Imageboards

open System
open OpenQA.Selenium
open System.Linq
open System.Text
open System.Text.RegularExpressions
open System.Web
open System.Text.Json
open LmgRecap.Shared
open OpenQA.Selenium.Firefox

type MegucaHandler() =
    interface IImageboardHandler with
        member this.FetchThreadJson(driver, builder, maxId, filters: string array) =
            driver.Navigate().GoToUrl($"https://meta.4chan.gay/tech/{builder.ThreadId}")

            driver.FindElement(By.Id("post-data")).GetAttribute("textContent")
            |> JsonSerializer.Deserialize<MegucaThread>
            |> fun a -> a.posts
            |> Array.filter (fun p -> p.id > maxId)
            |> Array.map MegucaPost
            |> Array.map (fun i -> (this :> IImageboardHandler).MapPostToChainNode(filters, i))

        member this.MapPostToChainNode(filters: string array, post) =
            match post with
            | FourChanPost _ -> failwith "MapPostToChainNode is not applicable for Meguca"
            | EightChanPost _ -> failwith "MapPostToChainNode is not applicable for Meguca"
            | MegucaPost p ->
                let getFileExtension (fileType: int) =
                    match fileType with
                    | 0 -> ".jpg"
                    | 1 -> ".png"
                    | 2 -> ".gif"
                    | 3 -> ".webm"
                    | 6 -> ".mp4"
                    | _ -> ""

                { id = p.id
                  timestamp = p.time
                  now =
                    DateTimeOffset
                        .FromUnixTimeSeconds(p.time)
                        .UtcDateTime.ToString("dd MMM yyyy (ddd) HH:mm:ss")
                  filtered = filters.Any(fun filter -> Regex.IsMatch(p.body.ToLower(), filter))
                  links = [||]
                  replies = [||]
                  ratings = [||]
                  comment = ""
                  unsanitized = p.body
                  context = None
                  rating = -1
                  reasoning = ""
                  filename =
                    match p.image with
                    | Some image -> Some(image.sha1 + getFileExtension image.file_type)
                    | None -> None
                  caption = None
                  label = None
                  confidence = None }

        member this.BuildReferences(chainmap, node) =
            let quoteIndicator = (this :> IImageboardHandler).QuoteIndicator

            { node with
                links =
                    Regex.Matches(
                        node.unsanitized,
                        quoteIndicator + @"(\d{" + node.id.ToString().Length.ToString() + "})"
                    )
                    |> Seq.map (fun m -> m.Groups[1].ToString() |> Int64.Parse)
                    |> Seq.filter (fun id -> chainmap.ContainsKey id)
                    |> Array.ofSeq
                replies =
                    chainmap.Values
                    |> Seq.filter (fun p -> p.unsanitized.Contains(node.id.ToString()))
                    |> Seq.map (fun p -> p.id)
                    |> Seq.toArray }

        member this.Sanitize(node, fetchTitleFromUrlAsync) =
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
                        node.unsanitized
                        |> HttpUtility.HtmlDecode
                        |> fun c -> c.Replace("https://arxiv.org/pdf", "https://arxiv.org/abs").Trim()
                        |> fun c ->
                            (Regex.Matches(
                                c,
                                @"https:\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?"
                             )
                             |> Seq.map (fun m ->
                                 m.Value, fetchTitleFromUrlAsync m.Value |> Result.defaultValue "Error fetching title")
                             |> Seq.fold
                                 (fun (newText: string) (url, title) -> newText.Replace(url, $"[{title}]({url})"))
                                 c) }
            | _ -> node

        member this.QuoteIndicator = @"\u003E\u003E"

        member this.GetDownloadLink(driver: FirefoxDriver, url: string) : IWebElement =
            driver.FindElement(By.CssSelector($"a[href='{url}']"))

        member this.GetImageUrl(filename: string) : string = $"/assets/images/src/{filename}"

        member this.GetThreadUrl(threadId: string) : string =
            $"https://meta.4chan.gay/tech/{threadId}"
