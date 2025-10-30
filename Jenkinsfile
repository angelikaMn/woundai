pipeline {
    agent any

    environment {
        DEPLOY_USER = 'woundai'
        DEPLOY_HOST = '<YOUR_STATIC_IP>'   // replace with your actual GCP VM static IP
        APP_DIR = '/home/woundai/app'
    }

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/angelikaMn/woundai.git'
            }
        }

        stage('Deploy to Server') {
            steps {
                sshagent (credentials: ['jenkins-ssh-key']) {
                    sh """
                    ssh -o StrictHostKeyChecking=no ${DEPLOY_USER}@${DEPLOY_HOST} '
                        cd ${APP_DIR} &&
                        git pull &&
                        source venv/bin/activate &&
                        pip install -r requirements.txt &&
                        deactivate &&
                        sudo systemctl restart woundai &&
                        sudo systemctl status woundai --no-pager
                    '
                    """
                    }
            }
        }
    }

    post {
        success {
            echo '✅ Deployment succeeded! Your app is live on woundflask.site'
        }
        failure {
            echo '❌ Deployment failed. Check Jenkins logs for details.'
        }
    }
}
