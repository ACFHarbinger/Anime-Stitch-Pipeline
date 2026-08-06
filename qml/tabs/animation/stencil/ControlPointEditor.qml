/*!
    \qmltype ControlPointEditor
    \inqmlmodule ImageToolkit.Tabs.Animation.Stencil
    \brief Interactive control-point editor overlay for stitch alignment.

    ControlPointEditor renders a source image with draggable circular
    control-point handles overlaid.  Moving a handle emits \c pointsMoved
    so the backend can recompute the ARAP mesh warp.

    \qmlproperty var ControlPointEditor::points
    List of \c {x, y} objects in image-pixel coordinates.

    \qmlproperty string ControlPointEditor::imagePath
    Absolute path to the image rendered underneath the control points.

    \qmlsignal ControlPointEditor::pointsMoved(var points)
    Emitted when any handle is released; carries the updated point list.
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../../components"
import "../../../"

Item {
    id: root

    property var points: []
    property string imagePath: ""

    signal pointsMoved(var points)

    // Background image
    Image {
        id: bgImage
        anchors.fill: parent
        source: root.imagePath ? "file://" + root.imagePath : ""
        fillMode: Image.PreserveAspectFit
        visible: source !== ""
    }

    // Placeholder when no image
    Rectangle {
        anchors.fill: parent
        color: Style.secondaryBackground
        visible: !bgImage.visible
        Text {
            anchors.centerIn: parent
            text: "No image loaded"
            color: Style.mutedText
            opacity: 0.5
        }
    }

    // Control-point handles
    Repeater {
        model: root.points.length
        delegate: Rectangle {
            id: handle
            property int ptIndex: index
            property real ptX: root.points[index] ? root.points[index].x : 0
            property real ptY: root.points[index] ? root.points[index].y : 0

            width: 18; height: 18; radius: 9
            color: Style.accent
            border.color: "white"
            border.width: 2
            opacity: 0.85
            x: ptX - width / 2
            y: ptY - height / 2

            Drag.active: dragArea.drag.active

            MouseArea {
                id: dragArea
                anchors.fill: parent
                drag.target: parent
                drag.minimumX: -handle.width / 2
                drag.minimumY: -handle.height / 2
                drag.maximumX: root.width - handle.width / 2
                drag.maximumY: root.height - handle.height / 2

                onReleased: {
                    var updated = root.points.slice()
                    updated[handle.ptIndex] = { x: handle.x + handle.width / 2, y: handle.y + handle.height / 2 }
                    root.points = updated
                    root.pointsMoved(updated)
                }
            }

            ToolTip.visible: dragArea.containsMouse
            ToolTip.text: "Point " + (index + 1)
        }
    }

    // Instructions overlay
    Text {
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottomMargin: 8
        text: "Drag handles to adjust control points"
        color: Style.mutedText
        font.pixelSize: 11
        visible: root.points.length > 0
    }
}
