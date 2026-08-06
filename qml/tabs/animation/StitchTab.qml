/*!
    \qmltype StitchTab
    \inqmlmodule ImageToolkit.Tabs.Animation
    \brief Interactive anime panorama stitching tab.

    StitchTab drives the AnimeStitchPipeline from the GUI layer.  It
    exposes directory selection, algorithm feature toggles, and a live
    log/progress view.  The heavy stitching work runs in a background
    thread; signals update progress and status.

    Backend object: \c mainBackend.stitchTab

    \qmlsignal StitchTab::qml_source_path_changed(string path)
    Emitted when the user selects a new source directory.

    Key slots:
    \list
      \li \c browse_input_directory_qml(currentPath) — open directory picker
      \li \c start_stitch_qml() — start the AnimeStitchPipeline
      \li \c cancel_stitch_qml() — interrupt in-progress stitching
    \endlist

    Key properties: \c is_stitching, \c status_text, \c progress, \c log_output
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../components"
import "../../"

Item {
    id: root

    readonly property var tab: mainBackend && mainBackend.stitchTab ? mainBackend.stitchTab : null

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 0
        spacing: 0

        // Header bar
        Rectangle {
            Layout.fillWidth: true
            height: 50
            color: Style.secondaryBackground
            border.color: Style.border

            RowLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 10

                Text { text: "Stitch"; color: Style.text; font.pixelSize: 18; font.bold: true }
                Item { Layout.fillWidth: true }

                Text {
                    text: tab ? tab.status_text : "Ready."
                    color: Style.mutedText
                    font.pixelSize: 12
                }
            }
        }

        // Main content area
        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            // Left panel — input + settings
            Rectangle {
                SplitView.preferredWidth: 320
                SplitView.minimumWidth: 260
                color: Style.secondaryBackground
                border.color: Style.border

                ScrollView {
                    anchors.fill: parent

                    ColumnLayout {
                        width: parent.parent.width
                        anchors.margins: 12
                        spacing: 12

                        // Source directory
                        GroupBox {
                            title: "Source Directory"
                            Layout.fillWidth: true
                            RowLayout {
                                TextField {
                                    id: sourceDirField
                                    Layout.fillWidth: true
                                    text: tab ? tab.source_dir : ""
                                    placeholderText: "Directory of frames to stitch..."
                                    background: Rectangle { color: Style.secondaryBackground; border.color: Style.border; radius: 4 }
                                    color: Style.text
                                    readOnly: true
                                }
                                AppButton {
                                    text: "Browse"
                                    onClicked: if (tab) tab.browse_input_directory_qml(sourceDirField.text)
                                }
                            }
                        }

                        // Algorithm toggles
                        GroupBox {
                            title: "Algorithm Features"
                            Layout.fillWidth: true
                            ColumnLayout {
                                spacing: 4
                                CheckBox { text: "LoFTR Dense Matching"; palette.windowText: Style.text; checked: true
                                           onCheckedChanged: if (tab) tab.use_loftr = checked }
                                CheckBox { text: "BiRefNet Foreground Masking"; palette.windowText: Style.text; checked: true
                                           onCheckedChanged: if (tab) tab.use_birefnet = checked }
                                CheckBox { text: "APAP Mesh Warping"; palette.windowText: Style.text; checked: true
                                           onCheckedChanged: if (tab) tab.use_apap = checked }
                                CheckBox { text: "ECC Sub-pixel Alignment"; palette.windowText: Style.text; checked: true
                                           onCheckedChanged: if (tab) tab.use_ecc = checked }
                                CheckBox { text: "BaSiC Luma Correction"; palette.windowText: Style.text; checked: true
                                           onCheckedChanged: if (tab) tab.use_basic = checked }
                                CheckBox { text: "Composite Foreground"; palette.windowText: Style.text; checked: true
                                           onCheckedChanged: if (tab) tab.composite_fg = checked }
                                CheckBox { text: "Poisson Seam Blend"; palette.windowText: Style.text
                                           onCheckedChanged: if (tab) tab.use_poisson = checked }
                            }
                        }

                        // Output settings
                        GroupBox {
                            title: "Output"
                            Layout.fillWidth: true
                            RowLayout {
                                TextField {
                                    id: outputPathField
                                    Layout.fillWidth: true
                                    text: tab ? tab.output_path : ""
                                    placeholderText: "Output file path (leave blank to auto-name)..."
                                    background: Rectangle { color: Style.secondaryBackground; border.color: Style.border; radius: 4 }
                                    color: Style.text
                                }
                                AppButton {
                                    text: "Browse"
                                    onClicked: if (tab) tab.browse_output_qml()
                                }
                            }
                        }

                        // Run/Cancel
                        AppButton {
                            text: (tab && tab.is_stitching) ? "Cancel" : "Start Stitch"
                            Layout.fillWidth: true
                            background: Rectangle {
                                color: (tab && tab.is_stitching) ? "#e74c3c" : Style.accent
                                radius: Style.borderRadius
                            }
                            onClicked: {
                                if (!tab) return
                                if (tab.is_stitching) tab.cancel_stitch_qml()
                                else tab.start_stitch_qml()
                            }
                        }
                    }
                }
            }

            // Right panel — output image + log
            ColumnLayout {
                SplitView.fillWidth: true
                spacing: 0

                // Output image preview
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#1a1a1a"
                    border.color: Style.border

                    Image {
                        anchors.fill: parent
                        anchors.margins: 4
                        source: (tab && tab.output_path) ? "file://" + tab.output_path : ""
                        fillMode: Image.PreserveAspectFit
                        visible: source !== ""
                    }

                    Text {
                        anchors.centerIn: parent
                        text: "Stitched output will appear here"
                        color: Style.mutedText
                        opacity: 0.5
                        visible: !(tab && tab.output_path)
                    }

                    ProgressBar {
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.margins: 4
                        visible: tab ? tab.is_stitching : false
                        value: tab ? tab.progress / 100.0 : 0
                    }
                }

                // Log
                Rectangle {
                    Layout.fillWidth: true
                    height: 180
                    color: "black"
                    border.color: Style.border

                    ScrollView {
                        anchors.fill: parent
                        TextArea {
                            readOnly: true
                            text: tab ? tab.log_output : ""
                            color: "#00ff00"
                            font.family: "Monospace"
                            font.pixelSize: 11
                            background: null
                        }
                    }
                }
            }
        }
    }
}
