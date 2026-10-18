fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let command = args.get(1).cloned();
    let json = args.iter().any(|arg| arg == "--json");
    match cofferdb::cli::run_with_exit_code(args) {
        Ok((output, exit_code)) => {
            if json {
                print!("{output}");
            } else {
                print!("{}", cofferdb::cli::human_output(&output));
            }
            if exit_code != 0 {
                std::process::exit(exit_code);
            }
        }
        Err(err) => {
            if json {
                eprintln!("{}", cofferdb::cli::error_output(command.as_deref(), &err));
            } else {
                eprintln!("{}: {}", err.code().as_str(), err.message());
            }
            std::process::exit(error_exit_code(err.code()));
        }
    }
}

fn error_exit_code(code: cofferdb::ErrorCode) -> i32 {
    use cofferdb::ErrorCode;
    match code {
        ErrorCode::IoError
        | ErrorCode::FileFormatError
        | ErrorCode::ChecksumMismatch
        | ErrorCode::CorruptTail
        | ErrorCode::IndexCorruption
        | ErrorCode::RecoveryError
        | ErrorCode::ResourceExhausted => 2,
        ErrorCode::LockContention => 4,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cofferdb::ErrorCode;

    #[test]
    fn resource_exhaustion_uses_exit_two_without_changing_existing_categories() {
        for code in [
            ErrorCode::IoError,
            ErrorCode::FileFormatError,
            ErrorCode::ChecksumMismatch,
            ErrorCode::CorruptTail,
            ErrorCode::IndexCorruption,
            ErrorCode::RecoveryError,
            ErrorCode::ResourceExhausted,
        ] {
            assert_eq!(error_exit_code(code), 2);
        }
        assert_eq!(error_exit_code(ErrorCode::LockContention), 4);
        for code in [
            ErrorCode::InvalidJsonlSchema,
            ErrorCode::InvalidMetadataFilter,
            ErrorCode::InvalidSearchWeights,
            ErrorCode::InvalidMetadataBoost,
            ErrorCode::CliUsage,
        ] {
            assert_eq!(error_exit_code(code), 1);
        }
    }
}
