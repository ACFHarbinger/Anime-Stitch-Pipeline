use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dev_repo_template_rust::greet;

fn bench_greet(c: &mut Criterion) {
    c.bench_function("greet", |b| b.iter(|| greet(black_box("world"))));
}

criterion_group!(benches, bench_greet);
criterion_main!(benches);
