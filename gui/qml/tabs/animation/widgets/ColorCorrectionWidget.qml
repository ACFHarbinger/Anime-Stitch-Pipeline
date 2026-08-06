/*!
    \qmltype ColorCorrectionWidget
    \inqmlmodule ImageToolkit.Tabs.Animation.Widgets
    \brief Inline tone/color correction slider widget.

    ColorCorrectionWidget provides sliders for five per-image correction
    parameters: brightness, contrast, gamma, saturation, and hue.  All
    values are emitted together via \c valuesChanged whenever any slider
    moves.

    \qmlproperty real ColorCorrectionWidget::brightness
    Additive brightness offset in [-1, 1].  Default 0.

    \qmlproperty real ColorCorrectionWidget::contrast
    Contrast multiplier in [0, 3].  Default 1.

    \qmlproperty real ColorCorrectionWidget::gamma
    Gamma value in [0.1, 5].  Default 1.

    \qmlproperty real ColorCorrectionWidget::saturation
    Saturation multiplier in [0, 3].  Default 1.

    \qmlproperty real ColorCorrectionWidget::hue
    Hue rotation in degrees [-180, 180].  Default 0.

    \qmlsignal ColorCorrectionWidget::valuesChanged(real brightness, real contrast, real gamma, real saturation, real hue)
*/
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "../../../components"
import "../../../"

Rectangle {
    id: root

    property real brightness: 0.0
    property real contrast:   1.0
    property real gamma:      1.0
    property real saturation: 1.0
    property real hue:        0.0

    signal valuesChanged(real brightness, real contrast, real gamma, real saturation, real hue)

    color: Style.secondaryBackground
    border.color: Style.border
    radius: Style.borderRadius

    function _emit() {
        root.valuesChanged(root.brightness, root.contrast, root.gamma, root.saturation, root.hue)
    }

    GridLayout {
        anchors.fill: parent
        anchors.margins: 12
        columns: 3
        columnSpacing: 10
        rowSpacing: 6

        // Brightness
        Label { text: "Brightness:"; color: Style.text; Layout.preferredWidth: 90 }
        Slider {
            id: brightSlider
            Layout.fillWidth: true
            from: -100; to: 100; value: root.brightness * 100; stepSize: 1
            onMoved: { root.brightness = value / 100.0; root._emit() }
        }
        Text { text: (root.brightness >= 0 ? "+" : "") + (root.brightness * 100).toFixed(0); color: Style.mutedText; Layout.preferredWidth: 40 }

        // Contrast
        Label { text: "Contrast:"; color: Style.text }
        Slider {
            id: contrastSlider
            Layout.fillWidth: true
            from: 0; to: 300; value: root.contrast * 100; stepSize: 1
            onMoved: { root.contrast = value / 100.0; root._emit() }
        }
        Text { text: root.contrast.toFixed(2) + "×"; color: Style.mutedText; Layout.preferredWidth: 40 }

        // Gamma
        Label { text: "Gamma:"; color: Style.text }
        Slider {
            id: gammaSlider
            Layout.fillWidth: true
            from: 10; to: 500; value: root.gamma * 100; stepSize: 1
            onMoved: { root.gamma = value / 100.0; root._emit() }
        }
        Text { text: root.gamma.toFixed(2); color: Style.mutedText; Layout.preferredWidth: 40 }

        // Saturation
        Label { text: "Saturation:"; color: Style.text }
        Slider {
            id: satSlider
            Layout.fillWidth: true
            from: 0; to: 300; value: root.saturation * 100; stepSize: 1
            onMoved: { root.saturation = value / 100.0; root._emit() }
        }
        Text { text: root.saturation.toFixed(2) + "×"; color: Style.mutedText; Layout.preferredWidth: 40 }

        // Hue
        Label { text: "Hue:"; color: Style.text }
        Slider {
            id: hueSlider
            Layout.fillWidth: true
            from: -180; to: 180; value: root.hue; stepSize: 1
            onMoved: { root.hue = value; root._emit() }
        }
        Text { text: (root.hue >= 0 ? "+" : "") + root.hue.toFixed(0) + "°"; color: Style.mutedText; Layout.preferredWidth: 40 }

        // Reset button
        Item {}
        AppButton {
            text: "Reset All"
            Layout.fillWidth: true
            onClicked: {
                brightSlider.value = 0; root.brightness = 0
                contrastSlider.value = 100; root.contrast = 1.0
                gammaSlider.value = 100; root.gamma = 1.0
                satSlider.value = 100; root.saturation = 1.0
                hueSlider.value = 0; root.hue = 0
                root._emit()
            }
        }
        Item {}
    }
}
