source Desktop/Coding/ec2-protein-llm-embeddings/config.cfg

mkdir -p $OUTPUT_SAVE_DIR

# Naming the instance with the current date and time avoids the risk of 
# obtaining the ID of an old instance when looking up by name
INSTANCE_NAME="${INSTANCE_NAME}_$(date +%d-%m-%y_%H-%M)"

aws ec2 run-instances \
--image-id $AMI_ID \
--count 1 \
--instance-type $INSTANCE_TYPE \
--key-name $EC2_KEY \
--security-group-ids $SECURITY_GROUP \
--subnet-id $SUBNET_ID \
--tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value='"${INSTANCE_NAME}"'}]' \
--no-cli-pager

read INSTANCE_IPV4 INSTANCE_ID <<< $(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=${INSTANCE_NAME}" \
    --query 'Reservations[*].Instances[*].[PublicIpAddress,InstanceId]' \
    --output text)

echo "Launching instance..."

aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# This line may be displayed before the files begin to transfer, as this is not
# initiated immediately after instance-running is returned
echo "Instance launched. Copying files..."

scp -o StrictHostKeyChecking=no -r -i $KEY_PATH $TO_INSTANCE_DIR ec2-user@$INSTANCE_IPV4:~  || { echo "Failed to copy files. Interrupting script."; exit 1; }

echo "Files copied. Initiating embedding generation..."

ssh -o StrictHostKeyChecking=no -i $KEY_PATH ec2-user@$INSTANCE_IPV4 'bash ~/to_instance/setup_and_launch.sh' || { echo "Failed to launch embedding script. Interrupting script."; exit 1; }

echo "Embeddings generated. Copying output to local machine..."
scp -o StrictHostKeyChecking=no -i $KEY_PATH ec2-user@$INSTANCE_IPV4:'protein_embeddings.npz' "$OUTPUT_SAVE_DIR/protein_embeddings.npz" || { echo "Failed to copy embeddings. Interrupting script."; exit 1; }

echo "Output copied successfully. Terminating instace..."

aws ec2 terminate-instances --instance-ids $INSTANCE_ID || { echo "Failed to terminate instance. Please do so manually to avoid costs!"; exit 1; }

aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID

echo "Instance terminated successfully"