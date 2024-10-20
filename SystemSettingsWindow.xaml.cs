using System.Windows;

namespace AcaiFruitClasifier
{
    public partial class SystemSettingsWindow : Window
    {
        private string _configPath;

        public SystemSettingsWindow(string configPath)
        {
            InitializeComponent();
            _configPath = configPath;

            LoadCurrentSettings();
        }

        private void LoadCurrentSettings()
        {
            string windowMode = YamlManager.ReadYaml(_configPath, "WindowMode");

            if (bool.TryParse(windowMode, out bool value))
            {
                PlotInMainWindow.IsChecked = value;
                PlotInDifferentWindows.IsChecked = !value;
            }
            else
            {
                PlotInMainWindow.IsChecked = true;
                PlotInDifferentWindows.IsChecked = false;
            }
        }

        private void PlotInMainWindow_Checked(object sender, RoutedEventArgs e)
        {
            PlotInDifferentWindows.IsChecked = false;
        }

        private void PlotInDifferentWindows_Checked(object sender, RoutedEventArgs e)
        {
            PlotInMainWindow.IsChecked = false;
        }

        private void PlotInMainWindow_Unchecked(object sender, RoutedEventArgs e)
        {
            PlotInDifferentWindows.IsChecked = true;
        }

        private void PlotInDifferentWindows_Unchecked(object sender, RoutedEventArgs e)
        {
            PlotInMainWindow.IsChecked = true;
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            bool value = PlotInMainWindow.IsChecked == true;
            YamlManager.WriteYaml(_configPath, "WindowMode", value.ToString());
            this.Close();
        }
    }
}
