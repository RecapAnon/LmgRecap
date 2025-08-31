#nowarn 3261
namespace LmgRecap

module ResultExtensions =
    module Result =
        /// <summary>
        /// Converts a nullable value into a <c>Result</c>, using the given error if null
        ///
        /// Documentation is found here: <href>https://demystifyfp.gitbook.io/fstoolkit-errorhandling/fstoolkit.errorhandling/result/requirefunctions#requirenotnull</href>
        /// </summary>
        /// <param name="error">The error value to return if the value is null.</param>
        /// <param name="value">The nullable value to check.</param>
        /// <returns>An <c>Ok</c> result if the value is not null, otherwise an <c>Error</c> result with the specified error value.</returns>
        let inline requireNotNull (error: 'error) (value: 'ok | null) : Result<'ok, 'error> =
            match value with
            | Null -> Error error
            | nonnull -> Ok(Unchecked.nonNull nonnull)
