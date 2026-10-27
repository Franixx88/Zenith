//! Transparent UCI promotion-case adapter for unmodified legacy engines.
//! Usage: uci_normalize [ENGINE]. With no argument, read a sibling .engine file.
//! Only uppercase QRBN suffixes on coordinate tokens in bestmove/info are changed.
use std::{
    borrow::Cow,
    io::{self, BufRead, Write},
    path::PathBuf,
    process::{Child, Command, ExitCode, Stdio},
    thread,
};

fn normalize(line: &str) -> Cow<'_, str> {
    let mut tokens = line.split_ascii_whitespace();
    let command = tokens.next();
    if !matches!(command, Some("bestmove" | "info"))
        || (command == Some("info") && tokens.any(|t| t == "string"))
    {
        return Cow::Borrowed(line);
    }
    let bytes = line.as_bytes();
    let mut result = None;
    let mut start = 0;
    while start < bytes.len() {
        if bytes[start].is_ascii_whitespace() {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < bytes.len() && !bytes[end].is_ascii_whitespace() {
            end += 1;
        }
        let token = &bytes[start..end];
        if token.len() == 5
            && (b'a'..=b'h').contains(&token[0])
            && (b'1'..=b'8').contains(&token[1])
            && (b'a'..=b'h').contains(&token[2])
            && (b'1'..=b'8').contains(&token[3])
            && matches!(token[4], b'Q' | b'R' | b'B' | b'N')
        {
            result.get_or_insert_with(|| bytes.to_vec())[start + 4] = token[4].to_ascii_lowercase();
        }
        start = end;
    }
    result.map_or(Cow::Borrowed(line), |data| {
        Cow::Owned(String::from_utf8(data).expect("only ASCII substitutions"))
    })
}

struct Guard(Child);
impl Drop for Guard {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

fn run() -> Result<ExitCode, Box<dyn std::error::Error>> {
    let executable = std::env::current_exe()?;
    let engine = if let Some(path) = std::env::args_os().nth(1) {
        PathBuf::from(path)
    } else {
        let path = std::fs::read_to_string(executable.with_extension("engine"))?;
        executable.parent().unwrap().join(path.trim())
    };
    let mut child = Guard(
        Command::new(engine)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?,
    );
    let mut input = child.0.stdin.take().unwrap();
    let output = child.0.stdout.take().unwrap();
    thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if writeln!(input, "{line}")
                .and_then(|_| input.flush())
                .is_err()
            {
                return;
            }
            if line.trim() == "quit" {
                return;
            }
        }
        // Closing the GUI pipe also closes a legacy child cleanly.
        let _ = writeln!(input, "quit").and_then(|_| input.flush());
    });
    let mut target = io::stdout().lock();
    let mut changed = 0u64;
    for line in io::BufReader::new(output).lines() {
        let line = line?;
        let normalized = normalize(&line);
        changed += u64::from(matches!(normalized, Cow::Owned(_)));
        writeln!(target, "{normalized}")?;
        target.flush()?;
    }
    let status = child.0.wait()?;
    eprintln!("UCI promotion-case adapter: normalized {changed} protocol lines");
    Ok(if status.success() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn main() -> ExitCode {
    run().unwrap_or_else(|error| {
        eprintln!("UCI adapter: {error}");
        ExitCode::FAILURE
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changes_only_uppercase_promotion_suffixes_in_protocol_moves() {
        assert_eq!(
            normalize("bestmove b7b8Q ponder a2a1N"),
            "bestmove b7b8q ponder a2a1n"
        );
        assert_eq!(
            normalize("info depth 4 pv b7b8R a2a1B"),
            "info depth 4 pv b7b8r a2a1b"
        );
        assert_eq!(normalize("\tbestmove\tb7b8Q  \r"), "\tbestmove\tb7b8q  \r");
        for line in [
            "bestmove e2e4",
            "bestmove b7b8q",
            "bestmove 0000",
            "bestmove b7b8K",
            "bestmove B7b8Q",
            "bestmove b7b9Q",
            "bestmove b7b8Qx",
            "info string text b7b8Q",
            "id name b7b8Q",
            "position fen Q7",
        ] {
            assert!(matches!(normalize(line), Cow::Borrowed(_)), "{line}");
        }
    }
}
