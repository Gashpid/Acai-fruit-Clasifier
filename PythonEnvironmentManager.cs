using System.IO;
using System.Windows;
using System.Diagnostics;
using System.Windows.Controls;

public class PythonEnvironmentManager
{
    public string _venvPath = @"scripts\venv";
    private string _pythonExecutable;
    private bool _isVenv = false;

    public PythonEnvironmentManager()
    {
        _pythonExecutable = Path.Combine(_venvPath, "Scripts", "python.exe");
    }

    public void EnsureVirtualEnvironment()
    {
        if (!Directory.Exists(_venvPath))
        {
            string command = $"python -m venv \"{_venvPath}\"";
            ExecuteCommand(command, null);
            _isVenv = true;
        }
        else
        {
            _isVenv = false;
        }
    }

    public void CloneRepositoryAndInstallDependencies(TextBox consoleOutputTextBox)
    {
        if (_isVenv)
        {
            string repoUrl = "https://github.com/Gashpid/Acai-fruit-Clasifier.git";
            string localPath = Path.Combine(_venvPath, "repo");

            string cloneCommand = $"git clone \"{repoUrl}\" \"{localPath}\"";

            ExecuteCommandWithOutput(cloneCommand, consoleOutputTextBox);

            string absoluteLocalPath = Path.GetFullPath(_venvPath).Replace("\\", "/");
            string requirementsPath = Path.Combine(absoluteLocalPath, "repo/requirements.txt").Replace("\\", "/");

            string installCommand = $"cd {absoluteLocalPath + "/Scripts"} && python.exe -m pip install -r {requirementsPath}";

            ExecuteCommandWithOutput(installCommand, consoleOutputTextBox);
        }
    }
    private void ExecuteCommandWithOutput(string command, TextBox consoleOutput)
    {
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = "cmd.exe",
            Arguments = $"/C {command}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using (Process process = new Process())
        {
            process.StartInfo = startInfo;

            process.OutputDataReceived += (sender, args) =>
            {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        consoleOutput.AppendText(args.Data + Environment.NewLine);
                        consoleOutput.ScrollToEnd();
                    });
                }
            };

            process.ErrorDataReceived += (sender, args) =>
            {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        consoleOutput.AppendText(args.Data + Environment.NewLine);
                        consoleOutput.ScrollToEnd();
                    });
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();
        }
    }

    public void RunPythonScript(string scriptPath, string arguments, Action<string> onOutput, Action<string> onError)
    {
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = _pythonExecutable,
            Arguments = $"\"{scriptPath}\" {arguments}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using (Process process = new Process())
        {
            process.StartInfo = startInfo;
            process.OutputDataReceived += (sender, args) => onOutput(args.Data);
            process.ErrorDataReceived += (sender, args) => onError(args.Data);

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();
        }
    }

    public void StopVirtualEnvironment()
    {
        
    }
    public void ExecuteCommand(string command, string workingDirectory)
    {
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = "cmd.exe",
            Arguments = $"/C {command}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = workingDirectory
        };

        using (Process process = new Process())
        {
            process.StartInfo = startInfo;
            process.Start();
            process.WaitForExit();
        }
    }
}