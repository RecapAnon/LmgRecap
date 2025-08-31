namespace LmgRecap

open System.CommandLine

module CommandLineExtensions =
    type Command = System.CommandLine.Command
    type Argument<'T> = System.CommandLine.Argument<'T>
    type RootCommand = System.CommandLine.RootCommand

    let addAlias alias (command: Command) =
        command.AddAlias alias
        command

    let addCommand subCommand (command: RootCommand) =
        command.AddCommand subCommand
        command

    let addOption option (command: Command) =
        command.AddOption option
        command

    let addArgument argument (command: Command) =
        command.AddArgument argument
        command

    let setHandler1<'A> (handler: 'A -> unit) argument (command: Command) =
        command.SetHandler(handler, argument)
        command

    let setHandler2<'A, 'B> (handler: 'A -> 'B -> unit) argument1 argument2 (command: Command) =
        command.SetHandler(handler, argument1, argument2)
        command

    let setHandler3<'A, 'B, 'C> (handler: 'A -> 'B -> 'C -> unit) argument1 argument2 argument3 (command: Command) =
        command.SetHandler(handler, argument1, argument2, argument3)
        command

    let addGlobalOption option (command: RootCommand) =
        command.AddGlobalOption option
        command

    let invoke (argv: string array) (rc: RootCommand) = rc.Invoke argv
