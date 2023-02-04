$user_obj = Get-ChildItem Env:\USERNAME
$user = $user_obj.Value
Write-Host "Agent username is '$user'"
Write-Host "##vso[task.setvariable variable=agent_username;]$user"
