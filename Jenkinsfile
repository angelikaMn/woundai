pipeline {
    agent any

    environment {
        DEPLOY_HOST = '34.50.119.22'
        APP_DIR = '/home/woundai/app'
        VENV_DIR = '/home/woundai/venv'
        REPO_URL = 'https://github.com/angelikaMn/woundai.git'
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    credentialsId: 'github-credentials',
                    url: "${REPO_URL}"
            }
        }

        stage('Deploy to Server') {
            steps {
                sshagent(credentials: ['woundai']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no woundai@${DEPLOY_HOST} "
                            # -----------------------------
                            # 1. Prepare app directory
                            # -----------------------------
                            if [ ! -d ${APP_DIR} ]; then
                                echo '📦 Cloning repo for the first time...'
                                git clone ${REPO_URL} ${APP_DIR}
                            else
                                echo '🔁 Updating existing repository...'
                                cd ${APP_DIR} &&
                                git fetch origin main &&
                                git reset --hard origin/main
                            fi

                            # -----------------------------
                            # 2. Python virtual environment
                            # -----------------------------
                            if [ ! -d ${VENV_DIR} ]; then
                                echo '⚙️  Creating shared Python virtual environment...'
                                python3 -m venv ${VENV_DIR}
                            fi

                            # -----------------------------
                            # 3. Install dependencies (quiet + cached)
                            # -----------------------------
                            source ${VENV_DIR}/bin/activate &&
                            pip install -r ${APP_DIR}/requirements.txt --cache-dir ~/.cache/pip --quiet --no-deps &&
                            deactivate

                            # -----------------------------
                            # 4. Restart Flask service
                            # -----------------------------
                            echo '🔄 Restarting woundai service...'
                            sudo systemctl daemon-reload &&
                            sudo systemctl restart woundai

                            # -----------------------------
                            # 5. Verify status
                            # -----------------------------
                            if sudo systemctl is-active --quiet woundai; then
                                echo '✅ Deployment successful!'
                            else
                                echo '❌ Deployment failed!' && exit 1
                            fi
                        "
                    '''
                }
            }
        }
    }

    post {
        success {
            echo '✅ Deployment successful and service is live.'
        }
        failure {
            echo '❌ Deployment failed. Check Jenkins console for error logs.'
        }
    }
}
