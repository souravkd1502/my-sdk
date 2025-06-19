from flask import Blueprint, request, jsonify
from app.services.ssh_client import execute_command
from app.services.file_transfer import upload_file, download_file

bp = Blueprint('remote_exec', __name__)

@bp.route('/execute', methods=['POST'])
def execute():
    data = request.get_json()
    command = data.get('command')
    output, error = execute_command(command)
    return jsonify({'output': output, 'error': error})
