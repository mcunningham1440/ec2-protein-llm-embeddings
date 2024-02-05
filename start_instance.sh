source Desktop/Coding/llm_pipeline_parallel/config.cfg

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

aws ec2 wait instance-running --instance-ids $INSTANCE_ID

scp -r -i $KEY_PATH $TO_INSTANCE_DIR ec2-user@$INSTANCE_IPV4:~

ssh -i $KEY_PATH ec2-user@$INSTANCE_IPV4 'bash -s' < 'setup_and_launch.sh'