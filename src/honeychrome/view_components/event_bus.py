from PySide6.QtCore import QObject, Signal


class EventBus(QObject):
    ### experiment file actions
    newExpRequested = Signal() #new exp dialog
    newExpRequestedFromTemplate = Signal() #new exp dialog
    newExpRequestedFromThisTemplate = Signal() #new exp dialog from specified template
    openExpRequested = Signal() #open exp dialog
    saveAsTemplateRequested = Signal() #save as exp dialog
    # loadExpRequested can handle all experiment file create and load actions
    # new experiment --> (file_path, new=True, template=None)
    # new experiment from template --> (file_path, new=True, template=template_file_path)
    # load experiment from template --> (file_path, new=False, template=None)
    # another signal to handle autosaving of current experiment and save as template
    reloadExpRequested = Signal()
    loadExpRequested = Signal(str)
    saveExpRequested = Signal(str)
    autoSaveRequested = Signal()

    ### sample actions
    loadSampleRequested = Signal(str)
    showNewSampleWidget = Signal()
    batchAddSamples = Signal()
    newSampleRequested = Signal(str, str)
    sampleTreeUpdated = Signal()
    selectSample = Signal(str)
    showExportModal = Signal()
    batchExportRequested = Signal(str, bool)
    generateSampleReport = Signal()
    openImportFCSWidget = Signal(bool)

    ### view
    aboutHoneychrome = Signal()
    popupMessage = Signal(str)
    warningMessage = Signal(str)
    setMainWindowTitle = Signal(str)
    progress = Signal(int, int)
    statusMessage = Signal(str)
    # update oscilloscope

    ### instrument control
    findConnectInstrument = Signal()
    initialiseInstrument = Signal()
    startAcquisition = Signal()
    stopAcquisition = Signal()
    restartAcquisition = Signal()
    flushSip = Signal()
    backFlushSip = Signal()
    gainChanged = Signal(str, int) # change gains

    # update instrument configuration
    # update experiment preferences
    resetAxisReloadSample = Signal()

    ### cytometry plots
    modeChangeRequested = Signal(str)
    newPlotRequested = Signal(str, str) # creates new plot definition in experiment
    showNewPlot = Signal(str) # creates plot widget on grid widget
    plotChangeRequested = Signal(str, int) # updates relevant plot
    # change gating hierarchy - everything below and including gate name
    # emitters: gate inserted, deleted, renamed
    # connections: refresh gating widget, refresh title menus, update lookup tables, make list of relevant plots and recalculate
    updateSourceChildGates = Signal(str, str)
    updateChildGateLabelOffset = Signal(str, tuple)
    changedGatingHierarchy = Signal(str, str)
    axisTransformed = Signal(str)
    axesReset = Signal(list)
    histsStatsRecalculated = Signal(str, list)
    updateRois = Signal(str, int)

    ### spectral process
    showSelectedProfiles = Signal(list)
    spectralControlAdded = Signal()
    spectralModelUpdated = Signal()
    cleaningActivated = Signal(bool)   # True = cleaning UI active, False = hidden
    cleaningResultsReady = Signal()    # emitted after Clean Controls recalc finishes
    spectralProcessRefreshed = Signal()
    requestUpdateProcessHists = Signal()
    spilloverSelectedCellChanged = Signal(str, str)
    rawGateRenamed = Signal(str, str)   # (old_name, new_name) — propagate raw gate rename to spectral model
