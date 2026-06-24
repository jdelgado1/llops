targetScope = 'resourceGroup'

@description('Azure region for resources.')
param location string = resourceGroup().location

@description('Base name used for Azure resources.')
param name string = 'tmg-llmops'

@description('Container image to run. Replace after build/push, or let azd set this during deployment.')
param imageName string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

@description('Foundry project endpoint, e.g. https://...services.ai.azure.com/api/projects/proj-default')
param foundryProjectEndpoint string

@description('Azure OpenAI / AI Services account name for fine-tuning and deployments.')
param foundryAoaiAccount string = 'TuneSEfoundry'

@description('Resource group containing the Azure OpenAI / AI Services account.')
param foundryResourceGroup string = 'demo-rg'

@description('OneLake/Fabric workspace name.')
param onelakeWorkspace string = 'Fine Tune Demo'

@description('OneLake/Fabric lakehouse name.')
param onelakeLakehouse string = 'lh_llmops'

@description('Base fine-tune model.')
param baseFinetuneModel string = 'gpt-4.1-nano'

@description('Fine-tune training type accepted by the Azure OpenAI account endpoint.')
param finetuneTrainingType string = 'GlobalStandard'

@description('Teacher model/deployment name.')
param teacherModel string = 'gpt-5.4'

@description('Current student deployment name.')
param studentFinetunedDeployment string = 'gpt-41-nano-student-v1'

@description('Current baseline deployment name.')
param baselineDeployment string = 'gpt-41-nano-base'

var safeName = toLower(replace(name, '_', '-'))
var acrName = take('tmgllmops${uniqueString(resourceGroup().id)}', 50)

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${safeName}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${safeName}-mi'
  location: location
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${safeName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, identity.id, 'AcrPull')
  scope: acr
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}

resource job 'Microsoft.App/jobs@2024-03-01' = {
  name: '${safeName}-retrain-job'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    environmentId: env.id
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 7200
      replicaRetryLimit: 0
      registries: [
        {
          server: acr.properties.loginServer
          identity: identity.id
        }
      ]
      manualTriggerConfig: {
        parallelism: 1
        replicaCompletionCount: 1
      }
    }
    template: {
      containers: [
        {
          name: 'retrain-loop'
          image: imageName
          env: [
            { name: 'PYTHONPATH', value: '/app/src' }
            { name: 'FOUNDRY_PROJECT_ENDPOINT', value: foundryProjectEndpoint }
            { name: 'FOUNDRY_AOAI_ACCOUNT', value: foundryAoaiAccount }
            { name: 'FOUNDRY_RESOURCE_GROUP', value: foundryResourceGroup }
            { name: 'ONELAKE_WORKSPACE', value: onelakeWorkspace }
            { name: 'ONELAKE_LAKEHOUSE', value: onelakeLakehouse }
            { name: 'BASE_FINETUNE_MODEL', value: baseFinetuneModel }
            { name: 'FINETUNE_TRAINING_TYPE', value: finetuneTrainingType }
            { name: 'TEACHER_MODEL', value: teacherModel }
            { name: 'STUDENT_FINETUNED_DEPLOYMENT', value: studentFinetunedDeployment }
            { name: 'BASELINE_DEPLOYMENT', value: baselineDeployment }
            { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
          ]
          command: [ 'python' ]
          args: [ 'scripts/run_retrain_loop.py', '--once' ]
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
        }
      ]
    }
  }
}

output containerRegistryName string = acr.name
output containerRegistryLoginServer string = acr.properties.loginServer
output managedIdentityClientId string = identity.properties.clientId
output managedIdentityPrincipalId string = identity.properties.principalId
output containerAppsJobName string = job.name
