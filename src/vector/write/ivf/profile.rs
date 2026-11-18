use std::fs::read_to_string;
use std::process;
use std::time::Instant;

const MINOR_FAULTS_FIELD: usize = 10;
const MAJOR_FAULTS_FIELD: usize = 12;
const USER_CPU_FIELD: usize = 14;
const SYSTEM_CPU_FIELD: usize = 15;
const FIRST_STAT_FIELD: usize = 3;

struct ProcessCounters {
    cpu_ticks: u64,
    minor_faults: u64,
    major_faults: u64,
    read_bytes: u64,
    write_bytes: u64,
}

impl ProcessCounters {
    fn parse(stat: &str, io: &str) -> Option<Self> {
        let fields: Vec<_> = stat.rsplit_once(") ")?.1.split_whitespace().collect();
        let stat_value = |field: usize| fields.get(field - FIRST_STAT_FIELD)?.parse::<u64>().ok();
        let io_value = |key: &str| {
            io.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name == key)
                    .then(|| value.trim().parse::<u64>().ok())
                    .flatten()
            })
        };
        Some(Self {
            cpu_ticks: stat_value(USER_CPU_FIELD)? + stat_value(SYSTEM_CPU_FIELD)?,
            minor_faults: stat_value(MINOR_FAULTS_FIELD)?,
            major_faults: stat_value(MAJOR_FAULTS_FIELD)?,
            read_bytes: io_value("read_bytes")?,
            write_bytes: io_value("write_bytes")?,
        })
    }

    fn read() -> Option<Self> {
        Self::parse(
            &read_to_string("/proc/self/stat").ok()?,
            &read_to_string("/proc/self/io").ok()?,
        )
    }
}

pub(super) struct StageProfile {
    start: Instant,
    counters: Option<ProcessCounters>,
}

impl StageProfile {
    pub(super) fn start() -> Self {
        Self {
            start: Instant::now(),
            counters: ProcessCounters::read(),
        }
    }

    pub(super) fn finish(self, stage: &str, vectors: usize) {
        let wall_ms = self.start.elapsed().as_millis();
        if let (Some(before), Some(after)) = (self.counters, ProcessCounters::read()) {
            log::info!(target: "paradedb::ivf_build",
                "ivf_stage_profile pid={} stage={} vectors={} wall_ms={} cpu_ticks={} read_bytes={} write_bytes={} minor_faults={} major_faults={}",
                process::id(), stage, vectors, wall_ms,
                after.cpu_ticks.saturating_sub(before.cpu_ticks),
                after.read_bytes.saturating_sub(before.read_bytes),
                after.write_bytes.saturating_sub(before.write_bytes),
                after.minor_faults.saturating_sub(before.minor_faults),
                after.major_faults.saturating_sub(before.major_faults));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProcessCounters;

    #[test]
    fn parses_process_counters_with_spaces_and_parentheses_in_name() {
        let stat = "42 (postgres (worker)) R 1 2 3 4 5 6 17 8 19 10 23 29";
        let io = "rchar: 999\nread_bytes: 4096\nwrite_bytes: 8192\n";
        let counters = ProcessCounters::parse(stat, io).unwrap();
        assert_eq!(counters.cpu_ticks, 52);
        assert_eq!(counters.minor_faults, 17);
        assert_eq!(counters.major_faults, 19);
        assert_eq!(counters.read_bytes, 4096);
        assert_eq!(counters.write_bytes, 8192);
        assert!(ProcessCounters::parse("truncated", io).is_none());
        assert!(ProcessCounters::parse(stat, "read_bytes: invalid").is_none());
    }
}
