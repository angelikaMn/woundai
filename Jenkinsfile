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
    sshagent(credentials: ['woundai']) {
        sh '''
        ssh -o StrictHostKeyChecking=no woundai@34.50.119.22 "
            if [ ! -d /home/woundai/app/.git ]; then
                echo '📦 First-time setup: cloning repo...'
                rm -rf /home/woundai/app
                git clone https://github.com/angelikaMn/woundai.git /home/woundai/app
            fi &&
            cd /home/woundai/app &&
            git pull &&
            source venv/bin/activate &&
            pip install --upgrade-strategy only-if-needed -r requirements.txt &&
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
    success {
      echo "✅ Build and Deployment Successful."
    }
    failure {
      echo "❌ Deployment failed. Check Jenkins logs for details."
    }
  }
}
