# Deployment Plan: TMG LLMOps Retraining Pipeline on Azure Container Apps

Status: Ready for Validation

## Goal
Host the local Python retraining/evaluation orchestration scripts in Azure so the loop can run without VS Code or a developer workstation.

## Proposed Target
Azure Container Apps Jobs for scheduled and event-driven pipeline runs.

## Current State
- Python scripts run locally from PowerShell using `.venv` and `PYTHONPATH=src`.
- Azure OpenAI / Foundry hosts models, fine-tuning jobs, and deployments.
- Microsoft Fabric / OneLake stores eval outputs and Blu's retraining exports.
- The pipeline currently bridges OneLake files to Azure OpenAI fine-tuning jobs.

## Selected Architecture
- Azure Container Apps Environment
- Azure Container Apps Job: retrain-loop runner with manual trigger initially
- Azure Container Registry for the pipeline container image
- User-assigned managed identity for Azure OpenAI and OneLake/Fabric access
- Container Apps environment variables for non-secret configuration
- Log Analytics for logs and run history
- Later scheduler: Container Apps Job cron trigger after manual validation
- Later event trigger: optional Event Grid/queue trigger when Blu writes a new manifest

## Pipeline Responsibilities
1. Detect new `Files/llmops/foundry_exports/<version>/manifest.json` in OneLake.
2. Download the referenced `train.jsonl` / `eval.jsonl`.
3. Normalize/convert records into the accepted fine-tune format.
4. Upload files to the Azure OpenAI account endpoint.
5. Submit a GlobalStandard fine-tune job.
6. Monitor job to terminal state.
7. Deploy the fine-tuned model.
8. Run held-out eval.
9. Upload `eval_results_*.json` and `eval_details_*.jsonl` to OneLake for Blu.
10. Persist state so the next run continues from the correct model.

## Decisions
- Trigger style: manual ACA Job first; move to cron polling after demo validation.
- State: keep using pipeline state JSON, then move to OneLake state path `Files/llmops/state/retrain_loop_state.json` in the next code hardening pass.
- Promotion: deploy/eval automatically for demo candidates; dashboard/scorecard communicates whether the candidate improved.
- Fabric integration: ACA Job polls OneLake for new `foundry_exports/<version>/manifest.json`.

## Generated Artifacts
- `Dockerfile`
- `.dockerignore`
- `azure.yaml`
- `infra/main.bicep`
- `infra/main.parameters.json`

## Security Notes
- Prefer managed identity over secrets.
- Grant least privilege to Azure OpenAI/AI Services and Fabric/OneLake.
- Keep local `.env` values out of container image.

## Next Steps
1. Run Azure validation.
2. Grant the user-assigned managed identity access to Azure OpenAI / Foundry and Fabric / OneLake.
3. Build and push the container image.
4. Deploy the ACA Job.
5. Trigger one manual job run and verify logs/results.
