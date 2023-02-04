user_obj=Get-ChildItem Env:\USERNAME
user=$user_obj.Value
pwsh -Command "Write-Host "Agent username is '$user'""
pwsh -Command "Write-Host "##vso[task.setvariable variable=agent_username;]$user""
Write-Host "Agent username is '$user'"
Write-Host "##vso[task.setvariable variable=agent_username;]$user"