/*!
    \qmltype RenderPanel
    \inqmlmodule ImageToolkit.Tabs.Animation.Stencil
    \brief Render output panel with settings overlay.

    RenderPanel displays the final stitched image alongside basic render
    metadata (dimensions, file size, elapsed time).  A spinning indicator
    is shown while rendering is in progress.

    \qmlproperty string RenderPanel::imagePath
    Absolute path to the rendered output image.

    \qmlproperty bool RenderPanel::isRendering
    Set to \c true while the stitch worker is running; shows a busy indicator.

    \qmlproperty string RenderPanel::metaText
    One-line render metadata string (e.g. "4320×1080 · 3.2 MB · 14.2 s").
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../../components"
import "../../../"

Rectangle {
    id: root

    property string imagePath:  ""
    property bool   isRendering: false
    property string metaText:   ""

    color: "#111"
    border.color: Style.border
    radius: Style.borderRadius
    clip: true

    Image {
        id: outputImage
        anchors.fill: parent
        anchors.margins: 4
        source: root.imagePath ? "file://" + root.imagePath : ""
        fillMode: Image.PreserveAspectFit
        visible: source !== ""
    }

    // Placeholder
    Text {
        anchors.centerIn: parent
        text: "Rendered output will appear here"
        color: Style.mutedText
        opacity: 0.4
        visible: !root.isRendering && root.imagePath === ""
    }

    // Busy indicator
    BusyIndicator {
        anchors.centerIn: parent
        running: root.isRendering
        visible: root.isRendering
    }

    // Metadata bar
    Rectangle {
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        height: root.metaText ? 28 : 0
        color: "#aa000000"
        visible: root.metaText !== ""

        Text {
            anchors.centerIn: parent
            text: root.metaText
            color: "white"
            font.pixelSize: 11
        }
    }
}
