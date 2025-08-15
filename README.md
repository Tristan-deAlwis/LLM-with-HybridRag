# Setting up Environment
```
py -3.11 -m venv /path/to/new/virtual/environment
./[VENV]/Scripts/activate
```

# Running App
#### This will start http://localhost:5000/generate
```python app.py```
</br>
</br>
</br>
To start feeding prompts, go to terminal and specify the header and body
```commandline
$headers = @{
>>     "Content-Type" = "application/json"
>> }
```

```commandline
$body = @{
>>     "query" = "What is the capital of France?"
>> } | ConvertTo-Json
```

```commandline
Invoke-RestMethod -Uri "http://localhost:5000/generate" -Method Post -Headers $headers -Body $body
```