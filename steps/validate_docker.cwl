#!/usr/bin/env cwl-runner
cwlVersion: v1.0
class: CommandLineTool
label: Score predictions file

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
    - entryname: check_submission.py
      entry: |
        #!/usr/bin/env python
        import argparse
        import json
        import synapseclient
        import challengeutils.submission as utils
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--synapse_config", required=True, help="Submission File")
        parser.add_argument("-s", "--submission_id", required=True, help="Scoring results")

        args = parser.parse_args()
        syn = synapseclient.Synapse(configPath=args.synapse_config)
        syn.login()
        try:
            utils.validate_docker_submission(syn, args.submission_id)
            status = "VALID"
            errors = ""
        except ValueError:
            status = "INVALID"
            errors = "Submission should be a Docker image."
        result = {
            "validation_status": status,
            "validation_errors": errors
        }
        with open("results.json", "w") as o:
          o.write(json.dumps(result))

inputs:
  - id: submissionid
    type: int
  - id: synapse_config
    type: File

outputs:
  - id: results
    type: File
    outputBinding:
      glob: results.json
  - id: status
    type: string
    outputBinding:
      glob: results.json
      outputEval: $(JSON.parse(self[0].contents)['validation_status'])
      loadContents: true
  - id: invalid_reasons
    type: string
    outputBinding:
      glob: results.json
      outputEval: $(JSON.parse(self[0].contents)['validation_errors'])
      loadContents: true

baseCommand: python
arguments:
  - valueFrom: check_submission.py
  - valueFrom: $(inputs.submissionid)
    prefix: -s
  - valueFrom: $(inputs.synapse_config.path)
    prefix: -c

hints:
  DockerRequirement:
    dockerPull: sagebionetworks/challengeutils:v4.2.1

s:author:
- class: s:Person
  s:identifier: https://orcid.org/0000-0002-5622-7998
  s:email: verena.chung@sagebase.org
  s:name: Verena Chung

s:codeRepository: https://github.com/Sage-Bionetworks-Challenges/olfactory-mixtures-prediction
s:license: https://spdx.org/licenses/Apache-2.0

$namespaces:
  s: https://schema.org/
