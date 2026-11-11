pipeline {
    agent any

    // Manual only: leave job Build Triggers unchecked; do not add triggers here.

    parameters {
        string(
            name: 'PI_HOST',
            defaultValue: '192.168.50.167',
            description: 'Raspberry Pi hostname or IP'
        )
        string(
            name: 'PI_USER',
            defaultValue: 'king',
            description: 'SSH user on the Pi'
        )
        string(
            name: 'PI_REPO_PATH',
            defaultValue: '/home/king/Desktop/HamsterMonitor',
            description: 'Path to the HamsterMonitor git clone on the Pi'
        )
        string(
            name: 'DEPLOY_BRANCH',
            defaultValue: 'main',
            description: 'Branch to fast-forward on the Pi'
        )
        string(
            name: 'SSH_CREDENTIAL_ID',
            defaultValue: 'pi-ssh-key',
            description: 'Jenkins SSH username-with-private-key credential ID'
        )
    }

    options {
        disableConcurrentBuilds()
        timestamps()
    }

    stages {
        stage('Deploy Hardware') {
            steps {
                sshagent(credentials: [params.SSH_CREDENTIAL_ID]) {
                    sh """
                        set -e
                        ssh -o StrictHostKeyChecking=accept-new \
                            ${params.PI_USER}@${params.PI_HOST} \
                            "set -e
                             cd '${params.PI_REPO_PATH}'
                             git fetch --all
                             git checkout '${params.DEPLOY_BRANCH}'
                             git pull --ff-only origin '${params.DEPLOY_BRANCH}'
                             sudo systemctl restart hamster_monitor
                             sudo systemctl --no-pager status hamster_monitor
                             systemctl is-active hamster_monitor"
                    """
                }
            }
        }

        stage('Smoke Check') {
            steps {
                sh """
                    set -e
                    curl -fsS --max-time 15 "http://${params.PI_HOST}:8081/status" \
                        || curl -fsS --max-time 15 "http://${params.PI_HOST}:8081/"
                """
            }
        }
    }
}
