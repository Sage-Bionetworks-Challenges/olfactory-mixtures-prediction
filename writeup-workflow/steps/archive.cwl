#!/usr/bin/env cwl-runner
cwlVersion: v1.0
class: CommandLineTool
label: Archive a Project (writeup) submission

hints:
  DockerRequirement:
    dockerPull: sagebionetworks/challengeutils:v4.0.1

requirements:
  - class: InlineJavascriptRequirement

inputs:
  - id: submissionid
    type: int
  - id: admin
    type: string
  - id: synapse_config
    type: File
  - id: check_validation_finished
    type: boolean?

outputs:
  - id: results
    type: File
    outputBinding:
      glob: results.json

baseCommand: challengeutils
arguments:
  - valueFrom: $(inputs.synapse_config.path)
    prefix: -c
  - valueFrom: archive-project
  - valueFrom: $(inputs.submissionid)
  - valueFrom: $(inputs.admin)
  - valueFrom: results.json
    prefix: --output

s:author:
- class: s:Person
  s:identifier: https://orcid.org/0000-0002-5622-7998
  s:email: verena.chung@sagebase.org
  s:name: Verena Chung

s:codeRepository: https://github.com/Sage-Bionetworks-Challenges/olfactory-mixtures-prediction
s:license: https://spdx.org/licenses/Apache-2.0

$namespaces:
  s: https://schema.org/