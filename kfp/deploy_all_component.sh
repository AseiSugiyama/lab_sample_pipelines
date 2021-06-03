GCP_PROJECT_ID=your-sample-pipeline-project # Enter your GCP Project ID
GCP_GCR_ENDPOINT=us.gcr.io # Enter your GCR endpoint like `asia.gcr.io`

for COMPONENT in 'data_generator' 'evaluator' 'trainer' 'transform'
do 
    COMPONENT_DIR=$(pwd)/components/${COMPONENT}
    PYPROJECT_TOML=${COMPONENT_DIR}/pyproject.toml
    docker build --target production -t $(awk -F'[ ="]+' '$1 == "name" { print $2 }' ${PYPROJECT_TOML} | sed 's/_/-/g'):latest ${COMPONENT_DIR}
    IMAGE_NAME=$(awk -F'[ ="]+' '$1 == "name" { print $2 }' ${PYPROJECT_TOML} | sed 's/_/-/g')
    IMAGE_VERSION_TAG=v$(awk -F'[ ="]+' '$1 == "version" { print $2 }' ${PYPROJECT_TOML})
    GCR_IMAGE_NAME_VERSIONED=${GCP_GCR_ENDPOINT}/${GCP_PROJECT_ID}/${IMAGE_NAME}:${IMAGE_VERSION_TAG}
    docker tag ${IMAGE_NAME}:latest ${GCR_IMAGE_NAME_VERSIONED}
    docker push ${GCR_IMAGE_NAME_VERSIONED}
done
