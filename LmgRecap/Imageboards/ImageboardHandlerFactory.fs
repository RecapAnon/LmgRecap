namespace LmgRecap.Imageboards

open LmgRecap.Shared

type ImageboardHandlerFactory() =
    static member CreateHandler(website: Website) : IImageboardHandler =
        match website with
        | Website.FourChan -> FourChanHandler() :> IImageboardHandler
        | Website.EightChan -> EightChanHandler() :> IImageboardHandler
        | Website.Meguca -> MegucaHandler() :> IImageboardHandler
        | _ -> failwith "Unsupported website type"
