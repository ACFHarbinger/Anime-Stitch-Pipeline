/*!
    \qmltype CanvasLayoutInspectorDialog
    \inqmlmodule ImageToolkit.Tabs.Animation.Dialog
    \brief Inspector dialog for the stitching canvas layout.

    CanvasLayoutInspectorDialog shows the affine placement of each frame on
    the stitching canvas: index, tx/ty translation, and bounding box.

    \qmlproperty var CanvasLayoutInspectorDialog::model
    List model of frame items.  Each item should expose:
    \list
      \li \c index — frame number
      \li \c tx, \c ty — canvas translation (px)
      \li \c width, \c height — frame dimensions (px)
    \endlist
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../../components"
import "../../../"

Dialog {
    id: root

    property var model: null

    title: "Canvas Layout Inspector"
    width: 640
    height: 480
    modal: true

    standardButtons: Dialog.Close

    background: Rectangle { color: Style.background; border.color: Style.border; radius: Style.borderRadius }

    contentItem: ColumnLayout {
        spacing: 12

        Text {
            text: "Frame placements on the stitching canvas"
            color: Style.mutedText
            font.pixelSize: 12
        }

        // Column headers
        RowLayout {
            spacing: 0
            Repeater {
                model: ["Index", "tx (px)", "ty (px)", "Width", "Height"]
                Text {
                    text: modelData
                    color: Style.accent
                    font.bold: true
                    Layout.preferredWidth: 110
                    padding: 4
                }
            }
        }

        Rectangle { height: 1; Layout.fillWidth: true; color: Style.border }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: root.model
            clip: true
            spacing: 2

            delegate: Rectangle {
                width: parent ? parent.width : 0
                height: 36
                color: index % 2 === 0 ? Style.secondaryBackground : "transparent"
                radius: 3

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 4
                    spacing: 0
                    Repeater {
                        model: [
                            model.index !== undefined ? model.index : index,
                            model.tx !== undefined ? model.tx.toFixed(1) : "—",
                            model.ty !== undefined ? model.ty.toFixed(1) : "—",
                            model.width !== undefined ? model.width : "—",
                            model.height !== undefined ? model.height : "—"
                        ]
                        Text {
                            text: modelData
                            color: Style.text
                            Layout.preferredWidth: 110
                            padding: 4
                            font.pixelSize: 12
                        }
                    }
                }
            }
        }
    }
}
