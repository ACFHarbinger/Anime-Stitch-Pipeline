/*!
    \qmltype SeamPainterWidget
    \inqmlmodule ImageToolkit.Tabs.Animation.Widgets
    \brief Seam-path painting widget for manual seam override.

    SeamPainterWidget displays an image with a coloured polyline overlay
    tracing the active DP seam path.  The user can click to add waypoints,
    effectively overriding the automatic seam selection.  Emits
    \c seamEdited with the new path when the overlay changes.

    \qmlproperty string SeamPainterWidget::imagePath
    Absolute path to the composite frame image to display.

    \qmlproperty var SeamPainterWidget::seamPath
    List of \c {x, y} objects (image-pixel coordinates) describing the seam.

    \qmlsignal SeamPainterWidget::seamEdited(var newPath)
    Emitted when the user adds or removes a waypoint.
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../../components"
import "../../../"

Item {
    id: root

    property string imagePath: ""
    property var    seamPath:  []

    signal seamEdited(var newPath)

    Rectangle {
        anchors.fill: parent
        color: "#111"
        border.color: Style.border
        radius: Style.borderRadius
        clip: true

        // Background image
        Image {
            id: bgImage
            anchors.fill: parent
            anchors.margins: 2
            source: root.imagePath ? "file://" + root.imagePath : ""
            fillMode: Image.PreserveAspectFit
        }

        // Seam path canvas
        Canvas {
            id: seamCanvas
            anchors.fill: bgImage

            onPaint: {
                var ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)

                if (!root.seamPath || root.seamPath.length < 2) return

                ctx.strokeStyle = "#e74c3c"
                ctx.lineWidth = 2
                ctx.setLineDash([6, 3])

                ctx.beginPath()
                ctx.moveTo(root.seamPath[0].x, root.seamPath[0].y)
                for (var i = 1; i < root.seamPath.length; i++) {
                    ctx.lineTo(root.seamPath[i].x, root.seamPath[i].y)
                }
                ctx.stroke()

                // Draw waypoint dots
                ctx.fillStyle = "#e74c3c"
                for (var j = 0; j < root.seamPath.length; j++) {
                    ctx.beginPath()
                    ctx.arc(root.seamPath[j].x, root.seamPath[j].y, 4, 0, Math.PI * 2)
                    ctx.fill()
                }
            }

            // Click to add a waypoint
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    var updated = root.seamPath.slice()
                    updated.push({ x: mouseX, y: mouseY })
                    root.seamPath = updated
                    seamCanvas.requestPaint()
                    root.seamEdited(updated)
                }
            }
        }

        // Toolbar
        Rectangle {
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            height: 36
            color: "#aa000000"

            RowLayout {
                anchors.fill: parent
                anchors.margins: 6
                spacing: 8
                Text { text: "Seam Painter"; color: "white"; font.bold: true }
                Item { Layout.fillWidth: true }
                AppButton {
                    text: "Clear"
                    onClicked: { root.seamPath = []; seamCanvas.requestPaint(); root.seamEdited([]) }
                    implicitHeight: 24
                }
            }
        }

        // Placeholder
        Text {
            anchors.centerIn: parent
            text: "Load an image and click to place seam waypoints"
            color: Style.mutedText
            opacity: 0.4
            wrapMode: Text.Wrap
            horizontalAlignment: Text.AlignHCenter
            width: parent.width - 40
            visible: root.imagePath === ""
        }
    }

    // Repaint when path changes
    onSeamPathChanged: if (seamCanvas) seamCanvas.requestPaint()
}
