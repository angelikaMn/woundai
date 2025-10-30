pipeline {
  agent any

  environment {
    DEPLOY_HOST = "34.50.119.22"
    DEPLOY_USER = "woundai"
    APP_DIR = "/home/woundai/app"
    SSH_KEY_ID = "woundai"
  }

  stages {

    stage('Checkout Code') {
      steps {
        git branch: 'main',
            url: 'https://github.com/angelikaMn/woundai.git',
            credentialsId: 'github-credentials'
      }
    }

    stage('Deploy to Server') {
      steps {
        sshagent(credentials: [env.SSH_KEY_ID]) {
          sh '''
            ssh -o StrictHostKeyChecking=no ${DEPLOY_USER}@${DEPLOY_HOST} '
              cd ${APP_DIR} &&
              git pull &&
              source venv/bin/activate &&
              pip install --upgrade-strategy only-if-needed --cache-dir ~/.cache/pip -r requirements.txt &&
              deactivate &&
              sudo systemctl daemon-reload &&
              sudo systemctl restart woundai &&
              sudo systemctl is-active --quiet woundai && echo "✅ Deployment successful" || (echo "❌ Deployment failed" && exit 1)
            '
          '''
        }
      }
    }
  }

  post {
    success {
      echo "✅ Build and Deployment Successful."
    }
    failure {
      echo "❌ Deployment failed. Check Jenkins logs for details."
    }
  }
}
