namespace LmgRecap.Imageboards

open System
open OpenQA.Selenium
open System.Text
open System.Linq
open System.Text.RegularExpressions
open System.Web
open System.Text.Json
open LmgRecap.Shared
open OpenQA.Selenium.Firefox

type EightChanHandler() =
    interface IImageboardHandler with
        member this.FetchThreadJson(driver, builder, maxId, filters: string array) =
            driver.Navigate().GoToUrl($"https://8chan.moe/ais/res/{builder.ThreadId}.json")
            System.Threading.Thread.Sleep(2000)
            driver.FindElement(By.CssSelector("h1 > a")).Click()

            driver.FindElement(By.TagName("pre")).Text
            |> JsonSerializer.Deserialize<EightThread>
            |> fun a -> a.posts
            |> Array.filter (fun p -> p.postId > maxId)
            |> Array.map EightChanPost
            |> Array.map (fun i -> (this :> IImageboardHandler).MapPostToChainNode(filters, i))

        member this.MapPostToChainNode(filters: string array, post) =
            match post with
            | FourChanPost _ -> failwith "MapPostToChainNode is not applicable for 8chan"
            | EightChanPost p ->
                let getUnixTimestamp (datetimeString: string) : int64 =
                    let format = "yyyy-MM-ddTHH:mm:ss.fffZ"

                    match
                        DateTime.TryParseExact(
                            datetimeString,
                            format,
                            null,
                            Globalization.DateTimeStyles.AssumeUniversal
                        )
                    with
                    | (true, datetimeObject) ->
                        let epoch = DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)
                        int64 (datetimeObject - epoch).TotalSeconds
                    | (false, _) -> failwith "Invalid datetime format"

                let ts = getUnixTimestamp p.creation

                { id = p.postId
                  timestamp = ts
                  now =
                    DateTimeOffset
                        .FromUnixTimeSeconds(ts)
                        .UtcDateTime.ToString("dd MMM yyyy (ddd) HH:mm:ss")
                  filtered = filters.Any(fun filter -> Regex.IsMatch(p.markdown.ToLower(), filter))
                  links = [||]
                  replies = [||]
                  ratings = [||]
                  comment = ""
                  unsanitized = p.markdown
                  context = None
                  rating = -1
                  reasoning = ""
                  filename =
                    (if p.files.Length > 0 then
                         Some p.files[0].originalName
                     else
                         None)
                  caption = None
                  label = None
                  confidence = None }
            | MegucaPost _ -> failwith "MapPostToChainNode is not applicable for 8chan"

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
                        node.unsanitized.Replace("<br>", "\n")
                        |> stripTags
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

        member this.QuoteIndicator = ">>"

        member this.GetDownloadLink(driver: FirefoxDriver, url: string) : IWebElement =
            driver.FindElement(By.CssSelector($"a[href][download='{url}']"))

        member this.GetImageUrl(filename: string) : string = filename

        member this.GetThreadUrl(threadId: string) : string =
            $"https://8chan.moe/ais/res/{threadId}.html"
