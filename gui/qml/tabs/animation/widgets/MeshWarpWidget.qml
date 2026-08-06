/*!
    \qmltype MeshWarpWidget
    \inqmlmodule ImageToolkit.Tabs.Animation.Widgets
    \brief Mesh-warp visualization widget with grid overlay.

    MeshWarpWidget renders an image with a semi-transparent rectangular
    mesh grid overlaid, visualising the APAP warp grid used for parallax
    correction.

    \qmlproperty string MeshWarpWidget::imagePath
    Absolute path to the image to display.

    \qmlproperty int MeshWarpWidget::meshRows
    Number of horizontal mesh divisions.  Default 8.

    \qmlproperty int MeshWarpWidget::meshCols
    Number of vertical mesh divisions.  Default 8.
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../../components"
import "../../../"

Rectangle {
    id: root

    property string imagePath: ""
    property int    meshRows:  8
    property int    meshCols:  8

    color: "#111"
    border.color: Style.border
    radius: Style.borderRadius
    clip: true

    Image {
        id: bgImage
        anchors.fill: parent
        anchors.margins: 2
        source: root.imagePath ? "file://" + root.imagePath : ""
        fillMode: Image.PreserveAspectFit
        visible: source !== ""
    }

    // Mesh grid overlay
    Canvas {
        anchors.fill: bgImage
        visible: bgImage.visible
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            ctx.strokeStyle = "rgba(0,188,212,0.45)"
            ctx.lineWidth = 1

            var cw = width  / root.meshCols
            var ch = height / root.meshRows

            // Vertical lines
            for (var c = 0; c <= root.meshCols; c++) {
                ctx.beginPath()
                ctx.moveTo(c * cw, 0)
                ctx.lineTo(c * cw, height)
                ctx.stroke()
            }
            // Horizontal lines
            for (var r = 0; r <= root.meshRows; r++) {
                ctx.beginPath()
                ctx.moveTo(0, r * ch)
                ctx.lineTo(width, r * ch)
                ctx.stroke()
            }
        }
    }

    // Placeholder
    Text {
        anchors.centerIn: parent
        text: "No image loaded"
        color: Style.mutedText
        opacity: 0.4
        visible: !bgImage.visible
    }

    // Settings bar
    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 32
        color: "#aa000000"
        visible: bgImage.visible

        RowLayout {
            anchors.fill: parent
            anchors.margins: 4
            spacing: 8
            Label { text: "Rows:"; color: "white"; font.pixelSize: 11 }
            SpinBox { from: 2; to: 32; value: root.meshRows; onValueChanged: root.meshRows = value; implicitHeight: 24 }
            Label { text: "Cols:"; color: "white"; font.pixelSize: 11 }
            SpinBox { from: 2; to: 32; value: root.meshCols; onValueChanged: root.meshCols = value; implicitHeight: 24 }
        }
    }
}
