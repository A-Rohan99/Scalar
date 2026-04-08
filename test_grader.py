import httpx

base = 'http://localhost:7860'

r1 = httpx.post(f'{base}/reset')
print("Reset response:", r1.json())
sid = r1.json().get('session_id')
print("Got Session ID:", sid)

r2 = httpx.get(f'{base}/grader', headers={'X-Session-ID': sid})
print("Grader response:", r2.status_code, r2.text)
