pipeline {
    agent any

    environment {
        DEPLOY_HOST = '34.50.119.22'
        APP_DIR = '/home/woundai/app'
    }

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    credentialsId: 'github-credentials',
                    url: 'https://github.com/angelikaMn/woundai.git'
            }
        }

        stage('Deploy to Server') {
            steps {
                sshagent(credentials: ['woundai']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no woundai@${DEPLOY_HOST} "
                            if [ ! -d ${APP_DIR}/.git ]; then
                                echo '📦 First-time setup: cloning repo...'
                                rm -rf ${APP_DIR}
                                git clone https://github.com/angelikaMn/woundai.git ${APP_DIR}
                            fi &&
                            cd ${APP_DIR} &&
                            git pull &&
                            if [ ! -d venv ]; then
                                python3 -m venv venv
                            fi &&
                            source venv/bin/activate &&
                            pip install --upgrade-strategy only-if-needed --cache-dir ~/.cache/pip -r requirements.txt &&
                            deactivate &&
                            sudo systemctl daemon-reload &&
                            sudo systemctl restart woundai &&
                            sudo systemctl is-active --quiet woundai && echo '✅ Deployment successful' || (echo '❌ Deployment failed' && exit 1)
                        "
                    '''
                }
            }
        }
    }

    post {
        failure {
            echo "❌ Deployment failed. Check Jenkins logs for details."
        }
        success {
            echo "✅ Deployment successful and running."
        }
    }
}
