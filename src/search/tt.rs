//! Local TT for one thread, coherent atomic bucket snapshots for Lazy SMP.
//! All shared payload accesses are SeqCst atomics. A bucket sequence brackets
//! writes; readers accept only matching even versions. Contention is a miss,
//! never a torn score. No unsafe code or probabilistic key/data XOR validation.
use super::{Bound, Entry};
use crate::board::{Move, Piece};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering::SeqCst},
};

#[repr(C, align(64))]
struct Bucket {
    sequence: AtomicU64,
    words: [[AtomicU64; 3]; 4],
}
impl Bucket {
    fn new(entries: &[Entry]) -> Self {
        Self {
            sequence: AtomicU64::new(0),
            words: std::array::from_fn(|i| {
                let packed = pack(entries[i]).unwrap_or([0; 3]);
                std::array::from_fn(|j| AtomicU64::new(packed[j]))
            }),
        }
    }
    fn snapshot(&self) -> Option<[Entry; 4]> {
        let before = self.sequence.load(SeqCst);
        if before & 1 != 0 {
            return None;
        }
        let entries =
            std::array::from_fn(|i| unpack(self.words[i].each_ref().map(|word| word.load(SeqCst))));
        (self.sequence.load(SeqCst) == before).then_some(entries)
    }
    fn write(&self, new: Entry) -> bool {
        let Some(packed) = pack(new) else {
            return false;
        };
        let version = self.sequence.load(SeqCst);
        if version & 1 != 0
            || self
                .sequence
                .compare_exchange(version, version.wrapping_add(1), SeqCst, SeqCst)
                .is_err()
        {
            return false;
        }
        // Exclusive writer owns this bucket. No panicking/caller code in this region.
        let mut entries: [Entry; 4] =
            std::array::from_fn(|i| unpack(self.words[i].each_ref().map(|word| word.load(SeqCst))));
        let (index, replaced) = replacement(&mut entries, new);
        if let Some(index) = index {
            // replacement may preserve an old best move.
            let final_words = if entries[index].best == new.best {
                packed
            } else {
                pack(entries[index]).unwrap_or(packed)
            };
            for (word, value) in self.words[index].iter().zip(final_words) {
                word.store(value, SeqCst);
            }
        }
        self.sequence.store(version.wrapping_add(2), SeqCst);
        replaced
    }
    fn clear(&self) {
        // Only called while every search worker is joined.
        self.sequence.fetch_add(1, SeqCst);
        for words in &self.words {
            for word in words {
                word.store(0, SeqCst);
            }
        }
        self.sequence.fetch_add(1, SeqCst);
    }
}
enum Storage {
    Local(Vec<Entry>),
    Shared(Arc<[Bucket]>),
}
pub(super) struct Table {
    storage: Storage,
    mask: usize,
}
pub(super) struct Probe {
    pub entry: Option<Entry>,
    pub best: Option<Move>,
    pub key_hit: bool,
    pub collision: bool,
}
impl Table {
    pub fn new(bytes: usize) -> Self {
        let available = (bytes / (4 * std::mem::size_of::<Entry>())).max(1);
        let buckets = 1usize << (usize::BITS - 1 - available.leading_zeros());
        Self {
            storage: Storage::Local(vec![Entry::default(); buckets * 4]),
            mask: buckets - 1,
        }
    }
    pub fn shared_clone(&mut self) -> Self {
        if let Storage::Local(entries) = &self.storage {
            let buckets: Arc<[Bucket]> = entries.chunks_exact(4).map(Bucket::new).collect();
            self.storage = Storage::Shared(buckets);
        }
        let Storage::Shared(buckets) = &self.storage else {
            unreachable!()
        };
        Self {
            storage: Storage::Shared(Arc::clone(buckets)),
            mask: self.mask,
        }
    }
    pub fn make_local(&mut self) {
        if let Storage::Shared(buckets) = &self.storage {
            self.storage = Storage::Local(
                buckets
                    .iter()
                    .flat_map(|b| b.snapshot().unwrap_or_default())
                    .collect(),
            );
        }
    }
    pub fn clear(&mut self) {
        match &mut self.storage {
            Storage::Local(entries) => entries.fill(Entry::default()),
            Storage::Shared(buckets) => buckets.iter().for_each(Bucket::clear),
        }
    }
    pub fn probe(&self, key: u64, context: u64, halfmove: u16, extensions: u8) -> Probe {
        let index = key as usize & self.mask;
        match &self.storage {
            Storage::Local(entries) => probe(
                &entries[index * 4..index * 4 + 4],
                key,
                context,
                halfmove,
                extensions,
            ),
            Storage::Shared(buckets) => probe(
                &buckets[index].snapshot().unwrap_or_default(),
                key,
                context,
                halfmove,
                extensions,
            ),
        }
    }
    pub fn write(&mut self, new: Entry) -> bool {
        let index = new.key as usize & self.mask;
        match &mut self.storage {
            Storage::Local(entries) => replacement(&mut entries[index * 4..index * 4 + 4], new).1,
            Storage::Shared(buckets) => buckets[index].write(new),
        }
    }
    #[cfg(test)]
    pub fn iter(&self) -> std::vec::IntoIter<Entry> {
        match &self.storage {
            Storage::Local(entries) => entries.clone(),
            Storage::Shared(buckets) => buckets
                .iter()
                .flat_map(|b| b.snapshot().unwrap_or_default())
                .collect(),
        }
        .into_iter()
    }
}
fn probe(bucket: &[Entry], key: u64, context: u64, halfmove: u16, extensions: u8) -> Probe {
    let mut found: Option<Entry> = None;
    let mut move_entry: Option<Entry> = None;
    let mut key_hit = false;
    for &entry in bucket {
        if !entry.valid || entry.key != key {
            continue;
        }
        key_hit = true;
        if entry.best.is_some() && move_entry.is_none_or(|old| entry.depth > old.depth) {
            move_entry = Some(entry);
        }
        if entry.context == context
            && entry.halfmove == halfmove
            && entry.extensions_left == extensions
            && found.is_none_or(|old| {
                entry.depth > old.depth || (entry.depth == old.depth && entry.bound == Bound::Exact)
            })
        {
            found = Some(entry);
        }
    }
    Probe {
        entry: found,
        best: move_entry.and_then(|e| e.best),
        key_hit,
        collision: !key_hit && bucket.iter().all(|e| e.valid),
    }
}
fn replacement(bucket: &mut [Entry], mut new: Entry) -> (Option<usize>, bool) {
    if let Some(index) = bucket.iter().position(|old| {
        old.valid
            && old.key == new.key
            && old.context == new.context
            && old.halfmove == new.halfmove
            && old.extensions_left == new.extensions_left
    }) {
        let old = bucket[index];
        if new.best.is_none() {
            new.best = old.best;
        }
        if old.generation == new.generation
            && old.depth > new.depth + 3
            && new.bound != Bound::Exact
        {
            return (None, false);
        }
        bucket[index] = new;
        return (Some(index), false);
    }
    let index = bucket.iter().position(|e| !e.valid).unwrap_or_else(|| {
        bucket
            .iter()
            .enumerate()
            .min_by_key(|(_, old)| {
                i32::from(old.depth) + 4 * i32::from(old.bound == Bound::Exact)
                    - 8 * i32::from(new.generation.wrapping_sub(old.generation))
            })
            .unwrap()
            .0
    });
    let replaced = bucket[index].valid;
    bucket[index] = new;
    (Some(index), replaced)
}
// key + full reversible-history fingerprint + metadata. Only searchable clocks
// (<100) are stored. Unsupported metadata is rejected, never silently truncated.
fn pack(entry: Entry) -> Option<[u64; 3]> {
    if !entry.valid {
        return Some([0; 3]);
    }
    if entry.halfmove > 127
        || !(0..=255).contains(&entry.depth)
        || entry.extensions_left > 2
        || i16::try_from(entry.score).is_err()
    {
        return None;
    }
    let mv = entry
        .best
        .and_then(|mv| {
            if mv.from >= 64 || mv.to >= 64 || mv.from == mv.to {
                return None;
            }
            let promotion = match mv.promotion {
                None => 0,
                Some(Piece::Knight) => 1,
                Some(Piece::Bishop) => 2,
                Some(Piece::Rook) => 3,
                Some(Piece::Queen) => 4,
                _ => return None,
            };
            Some(u64::from(mv.from) | (u64::from(mv.to) << 6) | (promotion << 12))
        })
        .unwrap_or(0);
    let bound = match entry.bound {
        Bound::Exact => 0,
        Bound::Lower => 1,
        Bound::Upper => 2,
    };
    let meta = mv
        | ((entry.score as i16 as u16 as u64) << 15)
        | (u64::from(entry.halfmove) << 31)
        | ((entry.depth as u64) << 38)
        | (u64::from(entry.generation) << 46)
        | (bound << 54)
        | (u64::from(entry.extensions_left) << 56)
        | (1 << 58);
    Some([entry.key, entry.context, meta])
}
fn unpack([key, context, meta]: [u64; 3]) -> Entry {
    if meta & (1 << 58) == 0 || meta >> 59 != 0 {
        return Entry::default();
    }
    let bound = match (meta >> 54) & 3 {
        0 => Bound::Exact,
        1 => Bound::Lower,
        2 => Bound::Upper,
        _ => return Entry::default(),
    };
    let encoded = meta & 0x7fff;
    let best = if encoded == 0 {
        None
    } else {
        let promotion = match encoded >> 12 {
            0 => None,
            1 => Some(Piece::Knight),
            2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook),
            4 => Some(Piece::Queen),
            _ => return Entry::default(),
        };
        Some(Move {
            from: (encoded & 63) as u8,
            to: ((encoded >> 6) & 63) as u8,
            promotion,
        })
    };
    Entry {
        key,
        context,
        score: i32::from((meta >> 15) as u16 as i16),
        halfmove: ((meta >> 31) & 127) as u16,
        depth: ((meta >> 38) & 255) as i16,
        generation: ((meta >> 46) & 255) as u8,
        bound,
        extensions_left: ((meta >> 56) & 3) as u8,
        best,
        valid: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(key: u64, generation: u8, depth: i16) -> Entry {
        Entry {
            key,
            generation,
            depth,
            valid: true,
            ..Entry::default()
        }
    }
    #[test]
    fn four_colliding_keys_and_distinct_score_contexts_survive() {
        assert_eq!(std::mem::size_of::<Entry>(), 32);
        assert_eq!(std::mem::size_of::<Bucket>(), 128);
        for shared in [false, true] {
            let mut table = Table::new(128);
            if shared {
                let _ = table.shared_clone();
            }
            for key in 0..4 {
                table.write(entry(key, 0, if key == 0 { 20 } else { 8 }));
            }
            for key in 0..4 {
                assert!(table.probe(key, 0, 0, 0).entry.is_some());
            }
            assert!(table.probe(4, 0, 0, 0).collision);
            for (ctx, clock, ext) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                assert!(table.probe(0, ctx, clock, ext).entry.is_none());
            }
            let mut other = entry(0, 0, 9);
            other.context = 42;
            table.write(other);
            assert!(table.probe(0, 42, 0, 0).entry.is_some());
            assert!(table.probe(0, 0, 0, 0).entry.is_some());
            table.make_local();
            assert!(table.probe(0, 42, 0, 0).entry.is_some());
        }
    }
    #[test]
    fn replacement_prefers_aged_shallow_entries_and_wraps_age() {
        let mut table = Table::new(128);
        table.write(entry(0, 255, 2));
        for key in 1..4 {
            table.write(entry(key, 0, 20));
        }
        assert!(table.write(entry(4, 0, 10)));
        assert!(table.probe(0, 0, 0, 0).entry.is_none());
        for key in 1..=4 {
            assert!(table.probe(key, 0, 0, 0).entry.is_some());
        }
        table.clear();
        assert!(table.iter().all(|e| !e.valid));
    }
    #[test]
    fn packed_fields_roundtrip_and_invalid_ranges_are_rejected() {
        for promotion in [
            None,
            Some(Piece::Knight),
            Some(Piece::Bishop),
            Some(Piece::Rook),
            Some(Piece::Queen),
        ] {
            let e = Entry {
                key: u64::MAX,
                context: 0x123456789abcdef,
                score: -30127,
                halfmove: 99,
                depth: 122,
                generation: 255,
                bound: Bound::Upper,
                extensions_left: 2,
                best: Some(Move {
                    from: 48,
                    to: 57,
                    promotion,
                }),
                valid: true,
            };
            let decoded = unpack(pack(e).unwrap());
            assert_eq!(decoded.key, e.key);
            assert_eq!(decoded.context, e.context);
            assert_eq!(decoded.score, e.score);
            assert_eq!(decoded.halfmove, e.halfmove);
            assert_eq!(decoded.depth, e.depth);
            assert_eq!(decoded.generation, e.generation);
            assert_eq!(decoded.bound as u8, e.bound as u8);
            assert_eq!(decoded.extensions_left, e.extensions_left);
            assert_eq!(decoded.best, e.best);
        }
        let mut e = entry(0, 0, 1);
        e.halfmove = 128;
        assert!(pack(e).is_none());
        e.halfmove = 0;
        e.score = 32768;
        assert!(pack(e).is_none());
    }
    #[test]
    fn concurrent_publication_never_accepts_torn_metadata() {
        let mut table = Table::new(128 * 16);
        let mut handles = vec![];
        for worker in 0..4 {
            let mut shared = table.shared_clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..20000u64 {
                    let key = (i * 13 + worker) % 64;
                    let context = key ^ 0x0045_ab11_9988;
                    let e = Entry {
                        context,
                        halfmove: (key % 100) as u16,
                        score: key as i32 * 97 - 3000,
                        extensions_left: (key % 3) as u8,
                        ..entry(key, key as u8, (key % 120 + 1) as i16)
                    };
                    shared.write(e);
                    if let Some(actual) = shared
                        .probe(key, context, e.halfmove, e.extensions_left)
                        .entry
                    {
                        assert_eq!(actual.score, e.score);
                        assert_eq!(actual.depth, e.depth);
                        assert_eq!(actual.generation, e.generation);
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
