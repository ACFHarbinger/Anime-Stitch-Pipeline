/*!
    \qmltype HybridStitchPanel
    \inqmlmodule ImageToolkit.Tabs.Animation.Stencil
    \brief Algorithm-toggle panel for hybrid stitch mode.

    HybridStitchPanel exposes a column of feature-toggle checkboxes that
    control which pipeline stages are active in hybrid stitch mode, along
    with a preview-mode toggle.

    \qmlproperty bool HybridStitchPanel::useLoftr
    \qmlproperty bool HybridStitchPanel::useBirefnet
    \qmlproperty bool HybridStitchPanel::useApap
    \qmlproperty bool HybridStitchPanel::useEcc
    \qmlproperty bool HybridStitchPanel::useBasic
    \qmlproperty bool HybridStitchPanel::showPreview
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../../components"
import "../../../"

Rectangle {
    id: root

    property bool useLoftr:    true
    property bool useBirefnet: true
    property bool useApap:     true
    property bool useEcc:      true
    property bool useBasic:    true
    property bool showPreview: false

    signal settingsChanged()

    color: Style.secondaryBackground
    border.color: Style.border
    radius: Style.borderRadius

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        Text { text: "Hybrid Stitch Controls"; color: Style.text; font.bold: true }

        Rectangle { height: 1; Layout.fillWidth: true; color: Style.border }

        CheckBox {
            text: "LoFTR Dense Matching"
            palette.windowText: Style.text
            checked: root.useLoftr
            onCheckedChanged: { root.useLoftr = checked; root.settingsChanged() }
        }
        CheckBox {
            text: "BiRefNet Foreground Masking"
            palette.windowText: Style.text
            checked: root.useBirefnet
            onCheckedChanged: { root.useBirefnet = checked; root.settingsChanged() }
        }
        CheckBox {
            text: "APAP Mesh Warping"
            palette.windowText: Style.text
            checked: root.useApap
            onCheckedChanged: { root.useApap = checked; root.settingsChanged() }
        }
        CheckBox {
            text: "ECC Sub-pixel Alignment"
            palette.windowText: Style.text
            checked: root.useEcc
            onCheckedChanged: { root.useEcc = checked; root.settingsChanged() }
        }
        CheckBox {
            text: "BaSiC Luma Correction"
            palette.windowText: Style.text
            checked: root.useBasic
            onCheckedChanged: { root.useBasic = checked; root.settingsChanged() }
        }

        Rectangle { height: 1; Layout.fillWidth: true; color: Style.border }

        CheckBox {
            text: "Show Stitch Preview"
            palette.windowText: Style.accent
            checked: root.showPreview
            onCheckedChanged: { root.showPreview = checked; root.settingsChanged() }
        }

        Item { Layout.fillHeight: true }
    }
}
