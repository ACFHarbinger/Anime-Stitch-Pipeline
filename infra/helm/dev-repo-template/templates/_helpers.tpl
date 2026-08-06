{{- define "dev-repo-template.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "dev-repo-template.labels" -}}
app: {{ include "dev-repo-template.name" . }}
{{- end -}}
