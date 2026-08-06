// Placeholder Tauri entry point — see ../../README.md. Not yet wired to
// backend/'s AnimeStitchPipeline.

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running Anime-Stitch-Pipeline frontend");
}
