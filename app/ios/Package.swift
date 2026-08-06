// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "AnimeStitchPipeline",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "AnimeStitchPipeline", targets: ["anime_stitch_pipeline"])
    ],
    targets: [
        .target(name: "anime_stitch_pipeline", path: "anime_stitch_pipeline")
    ]
)
