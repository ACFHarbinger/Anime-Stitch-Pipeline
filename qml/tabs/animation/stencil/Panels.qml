/*!
    \qmltype Panels
    \inqmlmodule ImageToolkit.Tabs.Animation.Stencil
    \brief Side-by-side before/after stitch preview panels.

    Panels shows two image panels: the left displays the raw input frame
    and the right displays the stitched result.  A divider can be dragged
    to resize the split.

    \qmlproperty string Panels::beforePath
    Absolute path to the input / "before" image.

    \qmlproperty string Panels::afterPath
    Absolute path to the stitched / "after" image.
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../../components"
import "../../../"

Item {
    id: root

    property string beforePath: ""
    property string afterPath:  ""

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        // Before panel
        Rectangle {
            SplitView.preferredWidth: root.width / 2
            SplitView.minimumWidth: 100
            color: "#1a1a1a"
            border.color: Style.border

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 4
                spacing: 4

                Text { text: "Before"; color: Style.mutedText; font.pixelSize: 11; Layout.alignment: Qt.AlignHCenter }

                Image {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    source: root.beforePath ? "file://" + root.beforePath : ""
                    fillMode: Image.PreserveAspectFit
                    visible: source !== ""
                }

                Text {
                    Layout.alignment: Qt.AlignCenter
                    text: "No input image"
                    color: Style.mutedText
                    opacity: 0.4
                    visible: root.beforePath === ""
                }
            }
        }

        // After panel
        Rectangle {
            SplitView.fillWidth: true
            color: "#1a1a1a"
            border.color: Style.border

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 4
                spacing: 4

                Text { text: "After"; color: Style.mutedText; font.pixelSize: 11; Layout.alignment: Qt.AlignHCenter }

                Image {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    source: root.afterPath ? "file://" + root.afterPath : ""
                    fillMode: Image.PreserveAspectFit
                    visible: source !== ""
                }

                Text {
                    Layout.alignment: Qt.AlignCenter
                    text: "No stitched output yet"
                    color: Style.mutedText
                    opacity: 0.4
                    visible: root.afterPath === ""
                }
            }
        }
    }
}
