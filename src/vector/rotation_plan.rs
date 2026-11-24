use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock, Mutex};

use lru::LruCache;
use quant::cascade::{LayerSpec, QueryRotationPlan};

const MAX_PLANS: usize = 32;
// Expanded rotations use about 24 bytes per layer coordinate, plus small headers.
const MAX_LAYER_COORDINATES: usize = 128 * 1024;

static PLANS: LazyLock<RotationPlanCache> = LazyLock::new(RotationPlanCache::new);

#[derive(Eq, Hash, PartialEq)]
struct PlanKey {
    dim: usize,
    specs: Vec<LayerSpec>,
}

struct RotationPlanCache {
    entries: Mutex<PlanEntries>,
}

struct PlanEntries {
    plans: LruCache<PlanKey, Arc<QueryRotationPlan>>,
    layer_coordinates: usize,
}

impl RotationPlanCache {
    fn new() -> Self {
        Self {
            entries: Mutex::new(PlanEntries::new()),
        }
    }

    fn get(&self, dim: usize, specs: &[LayerSpec]) -> Arc<QueryRotationPlan> {
        let key = PlanKey {
            dim,
            specs: specs.to_vec(),
        };
        if let Some(plan) = self.entries.lock().unwrap().plans.get(&key) {
            return Arc::clone(plan);
        }
        let plan = Arc::new(QueryRotationPlan::new(dim, specs));
        self.entries.lock().unwrap().insert(key, plan)
    }
}

impl PlanEntries {
    fn new() -> Self {
        Self {
            plans: LruCache::new(NonZeroUsize::new(MAX_PLANS).unwrap()),
            layer_coordinates: 0,
        }
    }

    fn insert(&mut self, key: PlanKey, plan: Arc<QueryRotationPlan>) -> Arc<QueryRotationPlan> {
        if let Some(existing) = self.plans.get(&key) {
            return Arc::clone(existing);
        }
        let weight = key.dim.saturating_mul(key.specs.len());
        if weight > MAX_LAYER_COORDINATES {
            return plan;
        }
        while self.plans.len() == MAX_PLANS
            || self.layer_coordinates + weight > MAX_LAYER_COORDINATES
        {
            let (evicted, _) = self.plans.pop_lru().unwrap();
            self.layer_coordinates -= evicted.dim * evicted.specs.len();
        }
        self.layer_coordinates += weight;
        self.plans.put(key, Arc::clone(&plan));
        plan
    }
}

/// Retains only immutable transforms; query values and segment readers are never cached here.
pub(crate) fn cached_rotation_plan(dim: usize, specs: &[LayerSpec]) -> Arc<QueryRotationPlan> {
    PLANS.get(dim, specs)
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;

    use quant::cascade::{prepare_fp_query_with_plan, LayerKind, Rotation};

    use super::*;

    fn sign(seed: u64) -> LayerSpec {
        LayerSpec {
            kind: LayerKind::Sign,
            bits: 1,
            rotation: Rotation::SeededFhtChaCha8 { seed },
        }
    }

    #[test]
    fn shares_cold_misses_across_threads_and_retains_between_queries() {
        let cache = Arc::new(RotationPlanCache::new());
        let barrier = Arc::new(Barrier::new(8));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    cache.get(1024, &[sign(91)])
                })
            })
            .collect();
        let plans: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        for plan in &plans {
            assert!(Arc::ptr_eq(&plans[0], plan));
        }
        let weak = Arc::downgrade(&plans[0]);
        drop(plans);
        assert!(Arc::ptr_eq(
            &weak.upgrade().unwrap(),
            &cache.get(1024, &[sign(91)])
        ));
    }

    #[test]
    fn full_schedule_and_dimension_are_part_of_cache_key() {
        let cache = RotationPlanCache::new();
        let grid = LayerSpec {
            kind: LayerKind::Grid,
            bits: 2,
            ..sign(93)
        };
        let cases = [
            (64, vec![sign(93)]),
            (65, vec![sign(93)]),
            (64, vec![sign(94)]),
            (64, vec![grid]),
            (64, vec![LayerSpec { bits: 3, ..grid }]),
            (
                64,
                vec![LayerSpec {
                    rotation: Rotation::None,
                    ..sign(93)
                }],
            ),
            (64, vec![sign(93), grid]),
            (64, vec![grid, sign(93)]),
        ];
        let plans: Vec<_> = cases
            .iter()
            .map(|(dim, specs)| cache.get(*dim, specs))
            .collect();
        for (i, (dim, specs)) in cases.iter().enumerate() {
            assert!(Arc::ptr_eq(&plans[i], &cache.get(*dim, specs)));
            for other in &plans[..i] {
                assert!(!Arc::ptr_eq(&plans[i], other));
            }
        }
    }

    #[test]
    fn cached_rotations_produce_identical_query_coordinates() {
        let specs = [
            sign(95),
            LayerSpec {
                kind: LayerKind::Grid,
                bits: 4,
                ..sign(96)
            },
        ];
        for dim in [64, 1024, 1536] {
            let cached = cached_rotation_plan(dim, &specs);
            let uncached = QueryRotationPlan::new(dim, &specs);
            for offset in [0.0, 0.5] {
                let query: Vec<_> = (0..dim).map(|i| (i as f32 + offset).sin()).collect();
                let actual = prepare_fp_query_with_plan(&query, &cached, specs.len());
                let expected = prepare_fp_query_with_plan(&query, &uncached, specs.len());
                for layer in 0..specs.len() {
                    let bits = |values: &[f32]| {
                        values
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(bits(actual.layer(layer)), bits(expected.layer(layer)));
                }
            }
        }
    }

    #[test]
    fn evicts_least_recently_used_plans_at_entry_limit() {
        let mut cache = PlanEntries::new();
        let mut plans = Vec::new();
        for seed in 0..MAX_PLANS as u64 {
            let specs = vec![sign(seed)];
            let plan = Arc::new(QueryRotationPlan::new(64, &specs));
            plans.push(Arc::downgrade(&plan));
            cache.insert(PlanKey { dim: 64, specs }, plan);
        }
        cache
            .plans
            .get(&PlanKey {
                dim: 64,
                specs: vec![sign(0)],
            })
            .unwrap();
        let specs = vec![sign(MAX_PLANS as u64)];
        cache.insert(
            PlanKey {
                dim: 64,
                specs: specs.clone(),
            },
            Arc::new(QueryRotationPlan::new(64, &specs)),
        );
        assert!(plans[0].upgrade().is_some());
        assert!(plans[1].upgrade().is_none());
        assert_eq!(cache.plans.len(), MAX_PLANS);
        assert_eq!(cache.layer_coordinates, MAX_PLANS * 64);
    }

    #[test]
    fn bounds_retained_coordinates_and_bypasses_oversized_plans() {
        let mut cache = PlanEntries::new();
        let specs = vec![sign(97)];
        let plan = Arc::new(QueryRotationPlan::new(MAX_LAYER_COORDINATES, &specs));
        cache.insert(
            PlanKey {
                dim: MAX_LAYER_COORDINATES,
                specs: specs.clone(),
            },
            Arc::clone(&plan),
        );
        let oversized = Arc::new(QueryRotationPlan::new(MAX_LAYER_COORDINATES + 1, &specs));
        let weak = Arc::downgrade(&oversized);
        drop(cache.insert(
            PlanKey {
                dim: MAX_LAYER_COORDINATES + 1,
                specs: specs.clone(),
            },
            oversized,
        ));
        assert!(weak.upgrade().is_none());
        assert_eq!(cache.layer_coordinates, MAX_LAYER_COORDINATES);
        let small = Arc::new(QueryRotationPlan::new(64, &specs));
        cache.insert(PlanKey { dim: 64, specs }, small);
        assert_eq!(cache.layer_coordinates, 64);
        assert_eq!(Arc::strong_count(&plan), 1);
        let mut values = vec![0.0; MAX_LAYER_COORDINATES];
        plan.rotate_layer_in_place(0, &mut values, &mut vec![0.0; MAX_LAYER_COORDINATES]);
        assert!(values.iter().all(|value| *value == 0.0));
    }
}
