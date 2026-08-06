/*!
    \qmltype StitchTrainTab
    \inqmlmodule ImageToolkit.Tabs.Models.Delta
    \brief AnimeStitchNet training sub-tab.

    StitchTrainTab trains the AnimeStitchNet registration model used by the
    Stitch pipeline for sub-pixel frame alignment.

    Backend object: \c mainBackend.stitchTrainTab

    Key slots: \c start_training(), \c stop_training(),
    \c browse_dataset_dir(), \c browse_output_dir()
    Key properties: \c is_training, \c status_text, \c progress, \c log_output
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../../components"
import "../../../"

Item {
    id: root

    readonly property var tab: mainBackend && mainBackend.stitchTrainTab ? mainBackend.stitchTrainTab : null

    ScrollView {
        anchors.fill: parent

        ColumnLayout {
            width: parent.parent.width
            anchors.margins: 20
            spacing: 15

            Text {
                text: "AnimeStitch Training"
                color: Style.text
                font.pixelSize: 22
                font.bold: true
            }

            GroupBox {
                title: "Dataset"
                Layout.fillWidth: true
                GridLayout {
                    columns: 3; columnSpacing: 10; rowSpacing: 8
                    Label { text: "Dataset Directory:"; color: Style.text }
                    TextField {
                        id: datasetField
                        Layout.fillWidth: true
                        text: tab ? tab.dataset_dir : ""
                        background: Rectangle { color: Style.secondaryBackground; border.color: Style.border; radius: 4 }
                        color: Style.text; readOnly: true
                    }
                    AppButton { text: "Browse"; onClicked: if (tab) tab.browse_dataset_dir() }

                    Label { text: "Output Directory:"; color: Style.text }
                    TextField {
                        id: outputField
                        Layout.fillWidth: true
                        text: tab ? tab.output_dir : ""
                        background: Rectangle { color: Style.secondaryBackground; border.color: Style.border; radius: 4 }
                        color: Style.text; readOnly: true
                    }
                    AppButton { text: "Browse"; onClicked: if (tab) tab.browse_output_dir() }
                }
            }

            GroupBox {
                title: "Training Parameters"
                Layout.fillWidth: true
                GridLayout {
                    columns: 2; columnSpacing: 16; rowSpacing: 8
                    Label { text: "Motion Model:"; color: Style.text }
                    ComboBox {
                        model: ["Translation", "Affine 4-DOF"]
                        onCurrentTextChanged: if (tab) tab.motion_model = currentText
                    }
                    Label { text: "Epochs:"; color: Style.text }
                    SpinBox { from: 1; to: 500; value: tab ? tab.epochs : 50
                              onValueChanged: if (tab) tab.epochs = value }
                    Label { text: "Batch Size:"; color: Style.text }
                    SpinBox { from: 1; to: 128; value: tab ? tab.batch_size : 8
                              onValueChanged: if (tab) tab.batch_size = value }
                    Label { text: "Learning Rate:"; color: Style.text }
                    TextField {
                        text: tab ? tab.lr : "1e-4"
                        background: Rectangle { color: Style.secondaryBackground; border.color: Style.border; radius: 4 }
                        color: Style.text
                        onTextChanged: if (tab) tab.lr = text
                        Layout.preferredWidth: 100
                    }
                    CheckBox {
                        text: "AMP (mixed precision)"
                        palette.windowText: Style.text
                        Layout.columnSpan: 2
                        onCheckedChanged: if (tab) tab.use_amp = checked
                    }
                }
            }

            // Mini loss chart (unicode sparkline rendered as monospace text)
            GroupBox {
                title: "Loss Chart"
                Layout.fillWidth: true
                Text {
                    text: tab ? tab.loss_sparkline : "▁▁▁▁▁▁▁▁"
                    color: Style.accent
                    font.family: "Monospace"
                    font.pixelSize: 14
                }
            }

            ProgressBar { Layout.fillWidth: true; value: tab ? tab.progress / 100.0 : 0; visible: tab ? tab.is_training : false }

            Rectangle {
                Layout.fillWidth: true; height: 150
                color: Style.secondaryBackground; border.color: Style.border; radius: Style.borderRadius
                ScrollView {
                    anchors.fill: parent
                    TextArea {
                        readOnly: true
                        text: tab ? tab.log_output : ""
                        color: Style.text; font.family: "Monospace"; font.pixelSize: 11
                        background: null
                    }
                }
            }

            AppButton {
                text: (tab && tab.is_training) ? "Stop Training" : "Start Training"
                Layout.fillWidth: true
                background: Rectangle { color: (tab && tab.is_training) ? "#e74c3c" : Style.accent; radius: Style.borderRadius }
                onClicked: { if (!tab) return; if (tab.is_training) tab.stop_training(); else tab.start_training() }
            }
        }
    }
}
