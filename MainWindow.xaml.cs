using Microsoft.Win32;
using System.IO;
using System.Windows;
using System.Windows.Threading;
using System.Diagnostics;

namespace AcaiFruitClasifier
{
    public partial class MainWindow : Window
    {
        private PythonEnvironmentManager _envManager = new PythonEnvironmentManager();
        private string _configPath = @"scripts\venv\repo\config.yaml";
        private string _htmlFilePath = "";
        private string _currentImage;
        private string _projectPath;

        public MainWindow()
        {
            InitializeComponent();

            this.Loaded += MainWindow_Loaded; 
        }

        private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            _projectPath = YamlManager.ReadYaml(_configPath, "ProjectPath");
            await InitializePythonEnvironmentAsync();
            InitializeWebView2();
        }

        private async Task InitializePythonEnvironmentAsync()
        {
            _envManager.EnsureVirtualEnvironment();
            await Task.Run(() => _envManager.CloneRepositoryAndInstallDependencies(ConsoleOutput));
        }

        private FileSystemWatcher _fileWatcher;

        private async void InitializeWebView2()
        {
            await webView2.EnsureCoreWebView2Async(null);

            UpdateHtmlFilePath(_htmlFilePath);
        }

        private void UpdateHtmlFilePath(string htmlFilePath)
        {
            _htmlFilePath = htmlFilePath;
            string appDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string fullHtmlFilePath = Path.Combine(appDirectory, _htmlFilePath);

            if (File.Exists(fullHtmlFilePath))
            {
                webView2.Source = new Uri($"file:///{fullHtmlFilePath.Replace("\\", "/")}");
            }

            if (_fileWatcher != null)
            {
                _fileWatcher.EnableRaisingEvents = false;
                _fileWatcher.Dispose();
            }

            StartMonitoringHtmlChanges(fullHtmlFilePath);
        }

        private void StartMonitoringHtmlChanges(string fullHtmlFilePath)
        {
            string directory = Path.GetDirectoryName(fullHtmlFilePath);
            string fileName = Path.GetFileName(fullHtmlFilePath);

            _fileWatcher = new FileSystemWatcher
            {
                Path = directory,
                Filter = fileName,
                NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.DirectoryName,
                EnableRaisingEvents = true
            };

            _fileWatcher.Changed += (s, e) =>
            {
                Dispatcher.Invoke(() =>
                {
                    webView2.Reload();
                });
            };

            _fileWatcher.Created += (s, e) =>
            {
                Dispatcher.Invoke(() =>
                {
                    webView2.Source = new Uri($"file:///{fullHtmlFilePath.Replace("\\", "/")}");
                });
            };

            _fileWatcher.Deleted += (s, e) =>
            {
                Dispatcher.Invoke(() =>
                {
                    ConsoleOutput.AppendText($"INFO: El archivo {fileName} fue eliminado." + Environment.NewLine);
                });
            };

            _fileWatcher.Renamed += (s, e) =>
            {
                Dispatcher.Invoke(() =>
                {
                    ConsoleOutput.AppendText($"INFO: El archivo fue renombrado a {e.FullPath}." + Environment.NewLine);
                    if (File.Exists(e.FullPath))
                    {
                        webView2.Source = new Uri($"file:///{e.FullPath.Replace("\\", "/")}");
                    }
                });
            };
        }

        private async void RunScript(string scriptPath)
        {
            await Task.Run(() =>
            {
                try
                {
                    _envManager.RunPythonScript(scriptPath, "", (output) =>
                    {
                        Dispatcher.Invoke(() => {
                            ConsoleOutput.AppendText(output + Environment.NewLine);
                            ConsoleOutput.ScrollToEnd();
                        });
                    }, (error) =>
                    {
                        Dispatcher.Invoke(() => {
                            ConsoleOutput.AppendText(error + Environment.NewLine);
                            ConsoleOutput.ScrollToEnd();
                        });
                    });
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => {
                        ConsoleOutput.AppendText(ex.Message + Environment.NewLine);
                        ConsoleOutput.ScrollToEnd();
                    });
                }
            });
        }

        private void OpenSystemSettings(object sender, RoutedEventArgs e)
        {
            SystemSettingsWindow settingsWindow = new SystemSettingsWindow(_configPath);
            settingsWindow.ShowDialog();
        }

        private void OpenFileWin(string title, string fileType, int mode)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Title = title,
                Filter = fileType,
                InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures)
            };

            if (openFileDialog.ShowDialog() == true)
            {
                string selectedFilePath = openFileDialog.FileName;
                if (mode == 0)
                {
                    string tag = "ModelPath";
                    YamlManager.WriteYaml(_configPath, tag, selectedFilePath);
                }
                else if (mode == 1)
                {
                    string tag = "CurrentImage";
                    _currentImage = selectedFilePath;
                    YamlManager.WriteYaml(_configPath, tag, _currentImage);
                }
                ConsoleOutput.AppendText($"Selected file: {selectedFilePath}" + Environment.NewLine);
            }
            else
            {
                ConsoleOutput.AppendText("No file selected." + Environment.NewLine);
            }
        }

        private async void SetProyectPath(object sender, RoutedEventArgs e)
        {
            using (var folderDialog = new System.Windows.Forms.FolderBrowserDialog())
            {
                folderDialog.RootFolder = Environment.SpecialFolder.Desktop;
                folderDialog.ShowNewFolderButton = true;

                System.Windows.Forms.DialogResult result = folderDialog.ShowDialog();

                if (result == System.Windows.Forms.DialogResult.OK && !string.IsNullOrWhiteSpace(folderDialog.SelectedPath))
                {
                    string selectedFolderPath = folderDialog.SelectedPath;
                    ConsoleOutput.AppendText($"Project folder: {selectedFolderPath}" + Environment.NewLine);
                    ConsoleOutput.ScrollToEnd();

                    string tag = "ProjectPath";
                    YamlManager.WriteYaml(_configPath, tag, selectedFolderPath);
                }
                else
                {
                    ConsoleOutput.AppendText("No folder selected." + Environment.NewLine);
                }
            }
        }
        private async void TrainModel(object sender, RoutedEventArgs e)
        {
            RunScript(@"scripts\venv\repo\train.py");
        }

        private async void TestModel(object sender, RoutedEventArgs e)
        {
            RunScript(@"scripts\venv\repo\test.py");
            string graphPath = Path.Combine(_projectPath, "output/predicted.html").Replace("\\", "/");
            UpdateHtmlFilePath(graphPath);
        }

        private async void LoadModel(object sender, RoutedEventArgs e)
        {
            OpenFileWin("Select a Model File", "Model files (*.keras, *.h5)|*.keras;*.h5", 0);
        }

        private void OpenImage(object sender, RoutedEventArgs e)
        {
            OpenFileWin("Select an Image File", "Image files (*.jpg, *.jpeg, *.png)|*.jpg;*.jpeg;*.png", 1);
        }

        private void PrepareDataset(object sender, RoutedEventArgs e)
        {
            RunScript(@"scripts\venv\repo\modules\prepare_dataset.py");
        }

        private void UtilsTones(object sender, RoutedEventArgs e)
        {
            string tag = "UtilsFucntion";
            string value = "Tone";

            YamlManager.WriteYaml(_configPath, tag, value);
            RunScript(@"scripts\venv\repo\modules\utils.py");
            string graphPath = Path.Combine(_projectPath, "output/tones.html").Replace("\\", "/");
            UpdateHtmlFilePath(graphPath);
        }
        private void UtilsIntensities(object sender, RoutedEventArgs e)
        {
            string tag = "UtilsFucntion";
            string value = "Intensities";

            YamlManager.WriteYaml(_configPath, tag, value);
            RunScript(@"scripts\venv\repo\modules\utils.py");
            string graphPath = Path.Combine(_projectPath, "output/intensities.html").Replace("\\", "/");
            UpdateHtmlFilePath(graphPath);
        }
        private void UtilsDiameter(object sender, RoutedEventArgs e)
        {
            string tag = "UtilsFucntion";
            string value = "Diameter";

            YamlManager.WriteYaml(_configPath, tag, value);
            RunScript(@"scripts\venv\repo\modules\utils.py");
            string graphPath = Path.Combine(_projectPath, "output/diameter.html").Replace("\\", "/");
            UpdateHtmlFilePath(graphPath);
        }
        private void UtilsCircularity(object sender, RoutedEventArgs e)
        {
            string tag = "UtilsFucntion";
            string value = "Circularity";

            YamlManager.WriteYaml(_configPath, tag, value);
            RunScript(@"scripts\venv\repo\modules\utils.py");
            string graphPath = Path.Combine(_projectPath, "output/circularity.html").Replace("\\", "/");
            UpdateHtmlFilePath(graphPath);
        }

        private void ExitMenuItem_Click(object sender, RoutedEventArgs e)
        {
            Application.Current.Shutdown();
        }

        private void OpenWebsiteButton_Click(object sender, RoutedEventArgs e)
        {
            string url = "https://github.com/Gashpid/Acai-fruit-Clasifier";
            OpenWebPage(url);
        }

        private void ClearConsole_Click(object sender, RoutedEventArgs e)
        {
            ConsoleOutput.Clear();
        }


        private void OpenWebPage(string url)
        {
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = url,
                    UseShellExecute = true
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error when opening web page: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }
}