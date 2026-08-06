// Placeholder entry point — see ../../../../../../README.md. Not yet wired
// to a backend/ API.
package com.acfharbinger.anime_stitch_pipeline

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Text

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Text("Anime-Stitch-Pipeline — scaffold, not yet implemented.")
        }
    }
}
