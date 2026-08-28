class I2C:
    def __init__(self, ft4222_communicator, base_address):
        self.ft4222 = ft4222_communicator
        self.base_address = base_address

        BaseAddress + 0x0000

    def write:

    def read:

    def write_read:

    def _write_read(self, address, write_buffer, write_size, read_buffer, read_size, caller_name):
        if ~self.ft4222.is_connected():
            return

        '''
        
    // Check if running when expected to be idle
	Status = DataLink->RegRead(BaseAddress + 0x0000);
	if (!(Status & 0x0010))
	{
		Log->Printf(MSGLOG_LVL_WARNING, MSGLOG_MASK_I2C, "I2C [Bus %s] %s: Unexpectedly running I2C core [0x%04X].", BusName, CallerName, Status);
	}

    // Flush the FIFOs
	DataLink->RegWrite(BaseAddress + 0x0000, 0x0002);
    // Verify the FIFO levels
    Status = DataLink->RegRead(BaseAddress + 0x0005);
    if (Status != 0)
	{
		Log->Printf(MSGLOG_LVL_WARNING, MSGLOG_MASK_I2C, "I2C [Bus %s] %s: In FIFO Level not cleared (%u).", BusName, CallerName, Status);
	}
    Status = DataLink->RegRead(BaseAddress + 0x0006);
    if (Status != 0)
	{
		Log->Printf(MSGLOG_LVL_WARNING, MSGLOG_MASK_I2C, "I2C [Bus %s] %s: Out FIFO Level not cleared (%u).", BusName, CallerName, Status);
	}

    // Set the device address
    DataLink->RegWrite(BaseAddress + 0x0001, Address >> 1);
    // Load the write buffer
    if (WriteBuffer)
	{
		for (Count=0; Count<WriteSize; Count++)
		{
			DataLink->RegWrite(BaseAddress + 0x0003, *WriteBuffer++);
		}
	}
    // Set the read size
    if (ReadBuffer)
	{
		DataLink->RegWrite(BaseAddress + 0x0002, ReadSize);
	}
	else
	{
		DataLink->RegWrite(BaseAddress + 0x0002, 0);
	}
    // Start the transfer
    DataLink->RegWrite(BaseAddress + 0x0000, 0x0001);
    // Wait for completion
    Count = 0;
    while (1)
    {
        Status = DataLink->RegRead(BaseAddress + 0x0000);
        if (Status & 0x0010)
		{
			Log->Printf(MSGLOG_LVL_MESSAGE, MSGLOG_MASK_I2C, "I2C [Bus %s] %s: Complete [0x%04X].", BusName, CallerName, Status);
			break;
		}
        Sleep(1);
        Count++;
        if (Count == 50)
		{
			Log->Printf(MSGLOG_LVL_WARNING, MSGLOG_MASK_I2C, "I2C [Bus %s] %s: Long wait detected [0x%04X].", BusName, CallerName, Status);
		}
		if (Count > 100000)
		{
			return false;
		}
    }

    // Check FIFO level vs. ReadSize
	Status = DataLink->RegRead(BaseAddress + 0x0006);
    if (ReadSize != Status)
	{
		Log->Printf(MSGLOG_LVL_WARNING, MSGLOG_MASK_I2C, "I2C [Bus %s] %s: Out data FIFO level mis-match (Exp:%u / Act:%u).", BusName, CallerName, ReadSize, Status);
	}
    // Unload the read buffer
    for (Count=0; Count<ReadSize; Count++)
    {
        *ReadBuffer++ = DataLink->RegRead(BaseAddress + 0x0004);
    }

    // Return success
    return true;
    '''