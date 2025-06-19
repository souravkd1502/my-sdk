import paramiko
from flask import current_app

def execute_command(command):
    if command not in current_app.config['ALLOWED_COMMANDS']:
        return '', 'Command not allowed.'
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='remote_host', username='user', password='pass')
    stdin, stdout, stderr = ssh.exec_command(command)
    output = stdout.read().decode()
    error = stderr.read().decode()
    ssh.close()
    return output, error
