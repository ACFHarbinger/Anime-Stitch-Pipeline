/*!
    \qmltype EdgeGraphInspectorDialog
    \inqmlmodule ImageToolkit.Tabs.Animation.Dialog
    \brief Inspector dialog for the frame-to-frame edge graph.

    EdgeGraphInspectorDialog shows all edges in the phase-correlation /
    feature-matching graph: source frame, destination frame, dx/dy offset,
    and confidence weight.

    \qmlproperty var EdgeGraphInspectorDialog::model
    List model of edge items.  Each item should expose:
    \list
      \li \c src — source frame index
      \li \c dst — destination frame index
      \li \c dx, \c dy — pixel offset
      \li \c weight — confidence score [0, 1]
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

    title: "Edge Graph Inspector"
    width: 700
    height: 500
    modal: true

    standardButtons: Dialog.Close

    background: Rectangle { color: Style.background; border.color: Style.border; radius: Style.borderRadius }

    contentItem: ColumnLayout {
        spacing: 12

        Text {
            text: "Phase-correlation / feature-matching edges between frames"
            color: Style.mutedText
            font.pixelSize: 12
        }

        RowLayout {
            spacing: 0
            Repeater {
                model: ["Src", "Dst", "dx (px)", "dy (px)", "Weight"]
                Text {
                    text: modelData
                    color: Style.accent
                    font.bold: true
                    Layout.preferredWidth: 120
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
                color: {
                    var w = model.weight !== undefined ? model.weight : 0
                    if (w >= 0.65) return "#1a2a1a"
                    if (w < 0.3)   return "#2a1a1a"
                    return index % 2 === 0 ? Style.secondaryBackground : "transparent"
                }
                radius: 3

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 4
                    spacing: 0
                    Repeater {
                        model: [
                            root.model ? root.model[index].src : "—",
                            root.model ? root.model[index].dst : "—",
                            root.model ? root.model[index].dx.toFixed(1) : "—",
                            root.model ? root.model[index].dy.toFixed(1) : "—",
                            root.model ? root.model[index].weight.toFixed(3) : "—"
                        ]
                        Text {
                            text: modelData
                            color: Style.text
                            Layout.preferredWidth: 120
                            padding: 4
                            font.pixelSize: 12
                        }
                    }
                }
            }
        }

        Text {
            text: "Green = high-confidence (≥0.65)  ·  Red = low-confidence (<0.30)"
            color: Style.mutedText
            font.pixelSize: 11
        }
    }
}
